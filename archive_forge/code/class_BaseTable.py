from __future__ import annotations
import datetime as dt
import uuid
from functools import partial
from types import FunctionType, MethodType
from typing import (
import numpy as np
import param
from bokeh.model import Model
from bokeh.models import ColumnDataSource, ImportedStyleSheet
from bokeh.models.widgets.tables import (
from bokeh.util.serialization import convert_datetime_array
from pyviz_comms import JupyterComm
from ..depends import transform_reference
from ..io.resources import CDN_DIST, CSS_URLS
from ..io.state import state
from ..reactive import Reactive, ReactiveData
from ..util import (
from ..util.warnings import warn
from .base import Widget
from .button import Button
from .input import TextInput
class BaseTable(ReactiveData, Widget):
    aggregators = param.Dict(default={}, nested_refs=True, doc="\n        A dictionary mapping from index name to an aggregator to\n        be used for hierarchical multi-indexes (valid aggregators\n        include 'min', 'max', 'mean' and 'sum'). If separate\n        aggregators for different columns are required the dictionary\n        may be nested as `{index_name: {column_name: aggregator}}`")
    editors = param.Dict(default={}, nested_refs=True, doc='\n        Bokeh CellEditor to use for a particular column\n        (overrides the default chosen based on the type).')
    formatters = param.Dict(default={}, nested_refs=True, doc='\n        Bokeh CellFormatter to use for a particular column\n        (overrides the default chosen based on the type).')
    hierarchical = param.Boolean(default=False, constant=True, doc='\n        Whether to generate a hierarchical index.')
    row_height = param.Integer(default=40, doc='\n        The height of each table row.')
    selection = param.List(default=[], doc='\n        The currently selected rows of the table.')
    show_index = param.Boolean(default=True, doc='\n        Whether to show the index column.')
    sorters = param.List(default=[], doc='\n        A list of sorters to apply during pagination.')
    text_align = param.ClassSelector(default={}, nested_refs=True, class_=(dict, str), doc="\n        A mapping from column name to alignment or a fixed column\n        alignment, which should be one of 'left', 'center', 'right'.")
    titles = param.Dict(default={}, nested_refs=True, doc='\n        A mapping from column name to a title to override the name with.')
    widths = param.ClassSelector(default={}, nested_refs=True, class_=(dict, int), doc='\n        A mapping from column name to column width or a fixed column\n        width.')
    value = param.Parameter(default=None)
    _data_params: ClassVar[List[str]] = ['value']
    _manual_params: ClassVar[List[str]] = ['formatters', 'editors', 'widths', 'titles', 'value', 'show_index']
    _rename: ClassVar[Mapping[str, str | None]] = {'hierarchical': None, 'name': None, 'selection': None}
    __abstract = True

    def __init__(self, value=None, **params):
        self._renamed_cols = {}
        self._filters = []
        self._index_mapping = {}
        self._edited_indexes = []
        super().__init__(value=value, **params)
        self.param.watch(self._setup_on_change, ['editors', 'formatters'])
        self.param.trigger('editors')
        self.param.trigger('formatters')

    @param.depends('value', watch=True, on_init=True)
    def _compute_renamed_cols(self):
        if self.value is None:
            self._renamed_cols.clear()
            return
        self._renamed_cols = {str(col) if str(col) != col else col: col for col in self._get_fields()}

    @property
    def _length(self):
        return len(self._processed)

    def _validate(self, *events: param.parameterized.Event):
        if self.value is None:
            return
        cols = self.value.columns
        if len(cols) != len(cols.drop_duplicates()):
            raise ValueError('Cannot display a pandas.DataFrame with duplicate column names.')

    def _get_fields(self) -> List[str]:
        indexes = self.indexes
        col_names = list(self.value.columns)
        if not self.hierarchical or len(indexes) == 1:
            col_names = indexes + col_names
        else:
            col_names = indexes[-1:] + col_names
        return col_names

    def _get_columns(self) -> List[TableColumn]:
        if self.value is None:
            return []
        indexes = self.indexes
        fields = self._get_fields()
        df = self.value.reset_index() if len(indexes) > 1 else self.value
        return self._get_column_definitions(fields, df)

    def _get_column_definitions(self, col_names: List[str], df: pd.DataFrame) -> List[TableColumn]:
        import pandas as pd
        indexes = self.indexes
        columns = []
        for col in col_names:
            if col in df.columns:
                data = df[col]
            elif col in self.indexes:
                if len(self.indexes) == 1:
                    data = df.index
                else:
                    data = df.index.get_level_values(self.indexes.index(col))
            if isinstance(data, pd.DataFrame):
                raise ValueError('DataFrame contains duplicate column names.')
            col_kwargs = {}
            kind = data.dtype.kind
            editor: CellEditor
            formatter: CellFormatter | None = self.formatters.get(col)
            if kind == 'i':
                editor = IntEditor()
            elif kind == 'b':
                editor = CheckboxEditor()
            elif kind == 'f':
                editor = NumberEditor()
            elif isdatetime(data) or kind == 'M':
                editor = DateEditor()
            else:
                editor = StringEditor()
            if col in self.editors and (not isinstance(self.editors[col], (dict, str))):
                editor = self.editors[col]
                if isinstance(editor, CellEditor):
                    editor = clone_model(editor)
            if col in indexes or editor is None:
                editor = CellEditor()
            if formatter is None or isinstance(formatter, (dict, str)):
                if kind == 'i':
                    formatter = NumberFormatter(text_align='right')
                elif kind == 'b':
                    formatter = StringFormatter(text_align='center')
                elif kind == 'f':
                    formatter = NumberFormatter(format='0,0.0[00000]', text_align='right')
                elif isdatetime(data) or kind == 'M':
                    if len(data) and isinstance(data.values[0], dt.date):
                        date_format = '%Y-%m-%d'
                    else:
                        date_format = '%Y-%m-%d %H:%M:%S'
                    formatter = DateFormatter(format=date_format, text_align='right')
                else:
                    formatter = StringFormatter()
                default_text_align = True
            else:
                if isinstance(formatter, CellFormatter):
                    formatter = clone_model(formatter)
                if hasattr(formatter, 'text_align'):
                    default_text_align = type(formatter).text_align.class_default(formatter) == formatter.text_align
                else:
                    default_text_align = True
            if not hasattr(formatter, 'text_align'):
                pass
            elif isinstance(self.text_align, str):
                formatter.text_align = self.text_align
                if not default_text_align:
                    msg = f"The 'text_align' in Tabulator.formatters[{col!r}] is overridden by Tabulator.text_align"
                    warn(msg, RuntimeWarning)
            elif col in self.text_align:
                formatter.text_align = self.text_align[col]
                if not default_text_align:
                    msg = f"The 'text_align' in Tabulator.formatters[{col!r}] is overridden by Tabulator.text_align[{col!r}]"
                    warn(msg, RuntimeWarning)
            elif col in self.indexes:
                formatter.text_align = 'left'
            if isinstance(self.widths, int):
                col_kwargs['width'] = self.widths
            elif str(col) in self.widths and isinstance(self.widths.get(str(col)), int):
                col_kwargs['width'] = self.widths.get(str(col))
            else:
                col_kwargs['width'] = 0
            title = self.titles.get(col, str(col))
            if col in indexes and len(indexes) > 1 and self.hierarchical:
                title = 'Index: %s' % ' | '.join(indexes)
            elif col in self.indexes and col.startswith('level_'):
                title = ''
            column = TableColumn(field=str(col), title=title, editor=editor, formatter=formatter, **col_kwargs)
            columns.append(column)
        return columns

    def _setup_on_change(self, *events: param.parameterized.Event):
        for event in events:
            self._process_on_change(event)

    def _process_on_change(self, event: param.parameterized.Event):
        old, new = (event.old, event.new)
        for model in (old if isinstance(old, dict) else {}).values():
            if not isinstance(model, (CellEditor, CellFormatter)):
                continue
            change_fn = self._editor_change if isinstance(model, CellEditor) else self._formatter_change
            for prop in model.properties() - Model.properties():
                try:
                    model.remove_on_change(prop, change_fn)
                except ValueError:
                    pass
        for model in (new if isinstance(new, dict) else {}).values():
            if not isinstance(model, (CellEditor, CellFormatter)):
                continue
            change_fn = self._editor_change if isinstance(model, CellEditor) else self._formatter_change
            for prop in model.properties() - Model.properties():
                model.on_change(prop, change_fn)

    def _editor_change(self, attr: str, new: Any, old: Any):
        self.param.trigger('editors')

    def _formatter_change(self, attr: str, new: Any, old: Any):
        self.param.trigger('formatters')

    def _update_index_mapping(self):
        if self._processed is None or (isinstance(self._processed, list) and (not self._processed)):
            self._index_mapping = {}
            return
        self._index_mapping = {i: index for i, index in enumerate(self._processed.index)}

    @updating
    def _update_cds(self, *events: param.parameterized.Event):
        old_processed = self._processed
        self._processed, data = self._get_data()
        self._update_index_mapping()
        if self.selection and old_processed is not None:
            indexes = list(self._processed.index)
            selection = []
            for sel in self.selection:
                try:
                    iv = old_processed.index[sel]
                    idx = indexes.index(iv)
                    selection.append(idx)
                except Exception:
                    continue
            self.selection = selection
        self._data = {k: _convert_datetime_array_ignore_list(v) for k, v in data.items()}
        msg = {'data': self._data}
        for ref, (m, _) in self._models.items():
            self._apply_update(events, msg, m.source, ref)

    def _process_param_change(self, params):
        if 'disabled' in params:
            params['editable'] = not params.pop('disabled') and len(self.indexes) <= 1
        params = super()._process_param_change(params)
        return params

    def _get_properties(self, doc: Document) -> Dict[str, Any]:
        properties = super()._get_properties(doc)
        properties['columns'] = self._get_columns()
        properties['source'] = cds = ColumnDataSource(data=self._data)
        cds.selected.indices = self.selection
        return properties

    def _get_model(self, doc: Document, root: Optional[Model]=None, parent: Optional[Model]=None, comm: Optional[Comm]=None) -> Model:
        properties = self._get_properties(doc)
        model = self._widget_type(**properties)
        root = root or model
        self._link_props(model.source, ['data'], doc, root, comm)
        self._link_props(model.source.selected, ['indices'], doc, root, comm)
        self._models[root.ref['id']] = (model, parent)
        return model

    def _update_columns(self, event: param.parameterized.Event, model: Model):
        if event.name == 'value' and [c.field for c in model.columns] == self._get_fields():
            return
        model.columns = self._get_columns()

    def _manual_update(self, events: Tuple[param.parameterized.Event, ...], model: Model, doc: Document, root: Model, parent: Optional[Model], comm: Optional[Comm]) -> None:
        for event in events:
            if event.type == 'triggered' and self._updating:
                continue
            elif event.name in ('value', 'show_index'):
                self._update_columns(event, model)
                if isinstance(model, DataCube):
                    model.groupings = self._get_groupings()
            elif hasattr(self, '_update_' + event.name):
                getattr(self, '_update_' + event.name)(model)
            else:
                self._update_columns(event, model)

    def _sort_df(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.sorters:
            return df
        fields = [self._renamed_cols.get(s['field'], s['field']) for s in self.sorters]
        ascending = [s['dir'] == 'asc' for s in self.sorters]
        df['_index_'] = np.arange(len(df)).astype(str)
        fields.append('_index_')
        ascending.append(True)
        if self.show_index:
            rename = 'index' in fields and df.index.name is None
            if rename:
                df.index.name = 'index'
        else:
            rename = False

        def tabulator_sorter(col):
            if col.dtype.kind not in 'SUO':
                return col
            try:
                return col.fillna('').str.lower()
            except Exception:
                return col
        df_sorted = df.sort_values(fields, ascending=ascending, kind='mergesort', key=tabulator_sorter)
        if rename:
            df.index.name = None
            df_sorted.index.name = None
        df.drop(columns=['_index_'], inplace=True)
        df_sorted.drop(columns=['_index_'], inplace=True)
        return df_sorted

    def _filter_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter the DataFrame.

        Parameters
        ----------
        df : DataFrame
           The DataFrame to filter

        Returns
        -------
        DataFrame
            The filtered DataFrame
        """
        filters = []
        for col_name, filt in self._filters:
            if col_name is not None and col_name not in df.columns:
                continue
            if isinstance(filt, (FunctionType, MethodType, partial)):
                df = filt(df)
                continue
            if isinstance(filt, param.Parameter):
                val = getattr(filt.owner, filt.name)
            else:
                val = filt
            column = df[col_name]
            if val is None:
                continue
            elif np.isscalar(val):
                mask = column == val
            elif isinstance(val, (list, set)):
                if not val:
                    continue
                mask = column.isin(val)
            elif isinstance(val, tuple):
                start, end = val
                if start is None and end is None:
                    continue
                elif start is None:
                    mask = column <= end
                elif end is None:
                    mask = column >= start
                else:
                    mask = (column >= start) & (column <= end)
            else:
                raise ValueError(f"'{col_name} filter value not understood. Must be either a scalar, tuple or list.")
            filters.append(mask)
        filters.extend(self._get_header_filters(df))
        if filters:
            mask = filters[0]
            for f in filters:
                mask &= f
            if self._edited_indexes:
                edited_mask = df.index.isin(self._edited_indexes)
                mask = mask | edited_mask
            df = df[mask]
        return df

    def _get_header_filters(self, df):
        filters = []
        for filt in getattr(self, 'filters', []):
            col_name = filt['field']
            op = filt['type']
            val = filt['value']
            filt_def = getattr(self, 'header_filters', {}) or {}
            if col_name in df.columns:
                col = df[col_name]
            elif col_name in self.indexes:
                if len(self.indexes) == 1:
                    col = df.index
                else:
                    col = df.index.get_level_values(self.indexes.index(col_name))
            else:
                continue
            if isinstance(val, list):
                if len(val) == 1:
                    val = val[0]
                elif not val:
                    continue
            val = col.dtype.type(val)
            if op == '=':
                filters.append(col == val)
            elif op == '!=':
                filters.append(col != val)
            elif op == '<':
                filters.append(col < val)
            elif op == '>':
                filters.append(col > val)
            elif op == '>=':
                filters.append(col >= val)
            elif op == '<=':
                filters.append(col <= val)
            elif op == 'in':
                if not isinstance(val, (list, np.ndarray)):
                    val = [val]
                filters.append(col.isin(val))
            elif op == 'like':
                filters.append(col.str.contains(val, case=False, regex=False))
            elif op == 'starts':
                filters.append(col.str.startsWith(val))
            elif op == 'ends':
                filters.append(col.str.endsWith(val))
            elif op == 'keywords':
                match_all = filt_def.get(col_name, {}).get('matchAll', False)
                sep = filt_def.get(col_name, {}).get('separator', ' ')
                matches = val.split(sep)
                if match_all:
                    for match in matches:
                        filters.append(col.str.contains(match, case=False, regex=False))
                else:
                    filt = col.str.contains(matches[0], case=False, regex=False)
                    for match in matches[1:]:
                        filt |= col.str.contains(match, case=False, regex=False)
                    filters.append(filt)
            elif op == 'regex':
                raise ValueError('Regex filtering not supported.')
            else:
                raise ValueError(f'Filter type {op!r} not recognized.')
        return filters

    def add_filter(self, filter, column=None):
        """
        Adds a filter to the table which can be a static value or
        dynamic parameter based object which will automatically
        update the table when changed..

        When a static value, widget or parameter is supplied the
        filtering will follow a few well defined behaviors:

          * scalar: Filters by checking for equality
          * tuple: A tuple will be interpreted as range.
          * list: A list will be interpreted as a set of discrete
                  scalars and the filter will check if the values
                  in the column match any of the items in the list.

        Arguments
        ---------
        filter: Widget, param.Parameter or FunctionType
            The value by which to filter the DataFrame along the
            declared column, or a function accepting the DataFrame to
            be filtered and returning a filtered copy of the DataFrame.
        column: str or None
            Column to which the filter will be applied, if the filter
            is a constant value, widget or parameter.

        Raises
        ------
        ValueError: If the filter type is not supported or no column
                    was declared.
        """
        if isinstance(filter, (tuple, list, set)) or np.isscalar(filter):
            deps = []
        elif isinstance(filter, (FunctionType, MethodType, partial)):
            deps = list(filter._dinfo['kw'].values()) if hasattr(filter, '_dinfo') else []
        else:
            filter = transform_reference(filter)
            if not isinstance(filter, param.Parameter):
                raise ValueError(f'{type(self).__name__} filter must be a constant value, parameter, widget or function.')
            elif column is None:
                raise ValueError('When filtering with a parameter or widget, a column to filter on must be declared.')
            deps = [filter]
        for dep in deps:
            dep.owner.param.watch(self._update_cds, dep.name)
        self._filters.append((column, filter))
        self._update_cds()

    def remove_filter(self, filter):
        """
        Removes a filter which was previously added.
        """
        self._filters = [(column, filt) for column, filt in self._filters if filt is not filter]
        self._update_cds()

    def _process_column(self, values):
        if not isinstance(values, (list, np.ndarray)):
            return [str(v) for v in values]
        if isinstance(values, np.ndarray) and values.dtype.kind == 'b':
            return values.tolist()
        return values

    def _get_data(self) -> Tuple[pd.DataFrame, DataDict]:
        return self._process_df_and_convert_to_cds(self.value)

    def _process_df_and_convert_to_cds(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, DataDict]:
        import pandas as pd
        df = self._filter_dataframe(df)
        if df is None:
            return ([], {})
        if isinstance(self.value.index, pd.MultiIndex):
            indexes = [f'level_{i}' if n is None else n for i, n in enumerate(df.index.names)]
        else:
            default_index = 'level_0' if 'index' in df.columns else 'index'
            indexes = [df.index.name or default_index]
        if len(indexes) > 1:
            df = df.reset_index()
        data = ColumnDataSource.from_df(df)
        if not self.show_index and len(indexes) > 1:
            data = {k: v for k, v in data.items() if k not in indexes}
        return (df, {k if isinstance(k, str) else str(k): self._process_column(v) for k, v in data.items()})

    def _update_column(self, column, array):
        import pandas as pd
        self.value[column] = array
        if self._processed is not None and self.value is not self._processed:
            with pd.option_context('mode.chained_assignment', None):
                self._processed[column] = array

    @property
    def indexes(self):
        import pandas as pd
        if self.value is None or not self.show_index:
            return []
        elif isinstance(self.value.index, pd.MultiIndex):
            return [f'level_{i}' if n is None else n for i, n in enumerate(self.value.index.names)]
        default_index = 'level_0' if 'index' in self.value.columns else 'index'
        return [self.value.index.name or default_index]

    def stream(self, stream_value, rollover=None, reset_index=True):
        """
        Streams (appends) the `stream_value` provided to the existing
        value in an efficient manner.

        Arguments
        ---------
        stream_value: (pd.DataFrame | pd.Series | Dict)
          The new value(s) to append to the existing value.
        rollover: int
           A maximum column size, above which data from the start of
           the column begins to be discarded. If None, then columns
           will continue to grow unbounded.
        reset_index: (bool, default=True)
          If True and the stream_value is a DataFrame,
          then its index is reset. Helps to keep the
          index unique and named `index`

        Raises
        ------
        ValueError: Raised if the stream_value is not a supported type.

        Examples
        --------

        Stream a Series to a DataFrame
        >>> value = pd.DataFrame({"x": [1, 2], "y": ["a", "b"]})
        >>> tabulator = Tabulator(value=value)
        >>> stream_value = pd.Series({"x": 4, "y": "d"})
        >>> tabulator.stream(stream_value)
        >>> tabulator.value.to_dict("list")
        {'x': [1, 2, 4], 'y': ['a', 'b', 'd']}

        Stream a Dataframe to a Dataframe
        >>> value = pd.DataFrame({"x": [1, 2], "y": ["a", "b"]})
        >>> tabulator = Tabulator(value=value)
        >>> stream_value = pd.DataFrame({"x": [3, 4], "y": ["c", "d"]})
        >>> tabulator.stream(stream_value)
        >>> tabulator.value.to_dict("list")
        {'x': [1, 2, 3, 4], 'y': ['a', 'b', 'c', 'd']}

        Stream a Dictionary row to a DataFrame
        >>> value = pd.DataFrame({"x": [1, 2], "y": ["a", "b"]})
        >>> tabulator = Tabulator(value=value)
        >>> stream_value = {"x": 4, "y": "d"}
        >>> tabulator.stream(stream_value)
        >>> tabulator.value.to_dict("list")
        {'x': [1, 2, 4], 'y': ['a', 'b', 'd']}

        Stream a Dictionary of Columns to a Dataframe
        >>> value = pd.DataFrame({"x": [1, 2], "y": ["a", "b"]})
        >>> tabulator = Tabulator(value=value)
        >>> stream_value = {"x": [3, 4], "y": ["c", "d"]}
        >>> tabulator.stream(stream_value)
        >>> tabulator.value.to_dict("list")
        {'x': [1, 2, 3, 4], 'y': ['a', 'b', 'c', 'd']}
        """
        import pandas as pd
        if not np.isfinite(self.value.index.max()):
            value_index_start = 1
        else:
            value_index_start = self.value.index.max() + 1
        if isinstance(stream_value, pd.DataFrame):
            if reset_index:
                stream_value = stream_value.reset_index(drop=True)
                stream_value.index += value_index_start
            combined = pd.concat([self.value, stream_value])
            if rollover is not None:
                combined = combined.iloc[-rollover:]
            with param.discard_events(self):
                self.value = combined
            try:
                self._updating = True
                self.param.trigger('value')
            finally:
                self._updating = False
            stream_value, stream_data = self._process_df_and_convert_to_cds(stream_value)
            try:
                self._updating = True
                self._stream(stream_data, rollover)
            finally:
                self._updating = False
        elif isinstance(stream_value, pd.Series):
            self.value.loc[value_index_start] = stream_value
            if rollover is not None and len(self.value) > rollover:
                with param.discard_events(self):
                    self.value = self.value.iloc[-rollover:]
            stream_value, stream_data = self._process_df_and_convert_to_cds(self.value.iloc[-1:])
            try:
                self._updating = True
                self._stream(stream_data, rollover)
            finally:
                self._updating = False
        elif isinstance(stream_value, dict):
            if stream_value:
                try:
                    stream_value = pd.DataFrame(stream_value)
                except ValueError:
                    stream_value = pd.Series(stream_value)
                self.stream(stream_value, rollover)
        else:
            raise ValueError('The stream value provided is not a DataFrame, Series or Dict!')

    def patch(self, patch_value, as_index=True):
        """
        Efficiently patches (updates) the existing value with the `patch_value`.

        Arguments
        ---------
        patch_value: (pd.DataFrame | pd.Series | Dict)
          The value(s) to patch the existing value with.
        as_index: boolean
          Whether to treat the patch index as DataFrame indexes (True)
          or as simple integer index.

        Raises
        ------
        ValueError: Raised if the patch_value is not a supported type.

        Examples
        --------

        Patch a DataFrame with a Dictionary row.
        >>> value = pd.DataFrame({"x": [1, 2], "y": ["a", "b"]})
        >>> tabulator = Tabulator(value=value)
        >>> patch_value = {"x": [(0, 3)]}
        >>> tabulator.patch(patch_value)
        >>> tabulator.value.to_dict("list")
        {'x': [3, 2], 'y': ['a', 'b']}

        Patch a Dataframe with a Dictionary of Columns.
        >>> value = pd.DataFrame({"x": [1, 2], "y": ["a", "b"]})
        >>> tabulator = Tabulator(value=value)
        >>> patch_value = {"x": [(slice(2), (3,4))], "y": [(1,'d')]}
        >>> tabulator.patch(patch_value)
        >>> tabulator.value.to_dict("list")
        {'x': [3, 4], 'y': ['a', 'd']}

        Patch a DataFrame with a Series. Please note the index is used in the update.
        >>> value = pd.DataFrame({"x": [1, 2], "y": ["a", "b"]})
        >>> tabulator = Tabulator(value=value)
        >>> patch_value = pd.Series({"index": 1, "x": 4, "y": "d"})
        >>> tabulator.patch(patch_value)
        >>> tabulator.value.to_dict("list")
        {'x': [1, 4], 'y': ['a', 'd']}

        Patch a Dataframe with a Dataframe. Please note the index is used in the update.
        >>> value = pd.DataFrame({"x": [1, 2], "y": ["a", "b"]})
        >>> tabulator = Tabulator(value=value)
        >>> patch_value = pd.DataFrame({"x": [3, 4], "y": ["c", "d"]})
        >>> tabulator.patch(patch_value)
        >>> tabulator.value.to_dict("list")
        {'x': [3, 4], 'y': ['c', 'd']}
        """
        if self.value is None:
            raise ValueError(f'Cannot patch empty {type(self).__name__}.')
        import pandas as pd
        if not isinstance(self.value, pd.DataFrame):
            raise ValueError(f'Patching an object of type {type(self.value).__name__} is not supported. Please provide a dict.')
        if isinstance(patch_value, pd.DataFrame):
            patch_value_dict = {column: list(patch_value[column].items()) for column in patch_value.columns}
            self.patch(patch_value_dict, as_index=as_index)
        elif isinstance(patch_value, pd.Series):
            if 'index' in patch_value:
                patch_value_dict = {k: [(patch_value['index'], v)] for k, v in patch_value.items()}
                patch_value_dict.pop('index')
            else:
                patch_value_dict = {patch_value.name: list(patch_value.items())}
            self.patch(patch_value_dict, as_index=as_index)
        elif isinstance(patch_value, dict):
            columns = list(self.value.columns)
            patches = {}
            for k, v in patch_value.items():
                values = []
                for patch_ind, value in v:
                    data_ind = patch_ind
                    if isinstance(patch_ind, slice):
                        data_ind = range(patch_ind.start, patch_ind.stop, patch_ind.step or 1)
                    if as_index:
                        if not isinstance(data_ind, range):
                            patch_ind = self.value.index.get_loc(patch_ind)
                            if not isinstance(patch_ind, int):
                                raise ValueError(f'Patching a table with duplicate index values is not supported. Found this duplicate index: {data_ind!r}')
                        self.value.loc[data_ind, k] = value
                    else:
                        self.value.iloc[data_ind, columns.index(k)] = value
                    if isinstance(value, pd.Timestamp):
                        value = datetime_as_utctimestamp(value)
                    elif value is pd.NaT:
                        value = BOKEH_JS_NAT
                    values.append((patch_ind, value))
                patches[k] = values
            self._patch(patches)
        else:
            raise ValueError(f'Patching with a patch_value of type {type(patch_value).__name__} is not supported. Please provide a DataFrame, Series or Dict.')

    @property
    def current_view(self):
        """
        Returns the current view of the table after filtering and
        sorting are applied.
        """
        df = self._processed
        return self._sort_df(df)

    @property
    def selected_dataframe(self):
        """
        Returns a DataFrame of the currently selected rows.
        """
        if not self.selection:
            return self.current_view.iloc[:0]
        return self.current_view.iloc[self.selection]