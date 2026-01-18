import pandas
from modin.core.dataframe.pandas.metadata import ModinIndex
from modin.error_message import ErrorMessage
from modin.utils import MODIN_UNNAMED_SERIES_LABEL, hashable
from .default2pandas.groupby import GroupBy, GroupByDefault
from .tree_reduce import TreeReduce
@classmethod
def _build_callable_for_dict(cls, agg_dict, preserve_aggregation_order=True, grp_has_id_level=False):
    """
        Build callable for an aggregation dictionary.

        Parameters
        ----------
        agg_dict : dict
            Aggregation dictionary.
        preserve_aggregation_order : bool, default: True
            Whether to manually restore the order of columns for the result specified
            by the `agg_func` keys (only makes sense when `agg_func` is a dictionary).
        grp_has_id_level : bool, default: False
            Whether the frame we're grouping on has intermediate columns
            (see ``cls.ID_LEVEL_NAME``).

        Returns
        -------
        callable(pandas.core.groupby.DataFrameGroupBy) -> pandas.DataFrame
        """
    from modin.pandas.utils import walk_aggregation_dict
    custom_aggs = {}
    native_aggs = {}
    result_columns = []
    for col, func, func_name, col_renaming_required in walk_aggregation_dict(agg_dict):
        dict_to_add = custom_aggs if cls.is_registered_implementation(func) else native_aggs
        new_value = func if func_name is None else (func_name, func)
        old_value = dict_to_add.get(col, None)
        if old_value is not None:
            ErrorMessage.catch_bugs_and_request_email(failure_condition=not isinstance(old_value, list), extra_log='Expected for all aggregation values to be a list when at least ' + f'one column has multiple aggregations. Got: {old_value} {type(old_value)}')
            old_value.append(new_value)
        else:
            dict_to_add[col] = [new_value] if col_renaming_required else new_value
        if col_renaming_required:
            func_name = str(func) if func_name is None else func_name
            result_columns.append((*(col if isinstance(col, tuple) else (col,)), func_name))
        else:
            result_columns.append(col)
    result_columns = pandas.Index(result_columns)

    def aggregate_on_dict(grp_obj, *args, **kwargs):
        """Aggregate the passed groupby object."""
        if len(native_aggs) == 0:
            native_agg_res = None
        elif grp_has_id_level:
            native_aggs_modified = {(cls.ID_LEVEL_NAME, *(key if isinstance(key, tuple) else (key,))): value for key, value in native_aggs.items()}
            native_agg_res = grp_obj.agg(native_aggs_modified)
            native_agg_res.columns = native_agg_res.columns.droplevel(0)
        else:
            native_agg_res = grp_obj.agg(native_aggs)
        custom_results = []
        insert_id_levels = False
        for col, func, func_name, col_renaming_required in walk_aggregation_dict(custom_aggs):
            if grp_has_id_level:
                cols_without_ids = grp_obj.obj.columns.droplevel(0)
                if isinstance(cols_without_ids, pandas.MultiIndex):
                    col_pos = cols_without_ids.get_locs(col)
                else:
                    col_pos = cols_without_ids.get_loc(col)
                agg_key = grp_obj.obj.columns[col_pos]
            else:
                agg_key = [col]
            result = func(grp_obj[agg_key])
            result_has_id_level = result.columns.names[0] == cls.ID_LEVEL_NAME
            insert_id_levels |= result_has_id_level
            if col_renaming_required:
                func_name = str(func) if func_name is None else func_name
                if result_has_id_level:
                    result.columns = pandas.MultiIndex.from_tuples([(old_col[0], col, func_name) for old_col in result.columns], names=[result.columns.names[0], result.columns.names[1], None])
                else:
                    result.columns = pandas.MultiIndex.from_tuples([(col, func_name)] * len(result.columns), names=[result.columns.names[0], None])
            custom_results.append(result)
        if insert_id_levels:
            for idx, ext_result in enumerate(custom_results):
                if ext_result.columns.names[0] != cls.ID_LEVEL_NAME:
                    custom_results[idx] = pandas.concat([ext_result], keys=[cls.ID_LEVEL_NAME], names=[cls.ID_LEVEL_NAME], axis=1, copy=False)
            if native_agg_res is not None:
                native_agg_res = pandas.concat([native_agg_res], keys=[cls.ID_LEVEL_NAME], names=[cls.ID_LEVEL_NAME], axis=1, copy=False)
        native_res_part = [] if native_agg_res is None else [native_agg_res]
        result = pandas.concat([*native_res_part, *custom_results], axis=1, copy=False)
        if preserve_aggregation_order and len(custom_aggs):
            result = result.reindex(result_columns, axis=1)
        return result
    return aggregate_on_dict