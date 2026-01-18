from collections import abc
import pandas
from pandas.core.dtypes.common import _get_dtype, is_bool_dtype
from .calcite_algebra import (
from .dataframe.utils import ColNameCodec
from .df_algebra import (
from .expr import (
def _process_groupby(self, op):
    """
        Translate ``GroupbyAggNode`` node.

        Parameters
        ----------
        op : GroupbyAggNode
            An operation to translate.
        """
    self.has_groupby = True
    frame = op.input[0]
    proj_cols = op.by.copy()
    for col in frame._table_cols:
        if col not in op.by:
            proj_cols.append(col)
    agg_exprs = op.agg_exprs
    cast_agg = self._bool_cast_aggregates
    if any((v.agg in cast_agg for v in agg_exprs.values())) and (bool_cols := {c: cast_agg[agg_exprs[c].agg] for c, t in frame.dtypes.items() if not isinstance(t, pandas.CategoricalDtype) and is_bool_dtype(t) and (agg_exprs[c].agg in cast_agg)}):
        trans = self._input_ctx()._maybe_copy_and_translate_expr
        proj_exprs = [trans(frame.ref(c).cast(bool_cols[c])) if c in bool_cols else self._ref(frame, c) for c in proj_cols]
    else:
        proj_exprs = [self._ref(frame, col) for col in proj_cols]
    compound_aggs = {}
    for agg, expr in agg_exprs.items():
        if expr.agg in self._compound_aggregates:
            compound_aggs[agg] = self._compound_aggregates[expr.agg](self, expr.operands)
            extra_exprs = compound_aggs[agg].gen_proj_exprs()
            proj_cols.extend(extra_exprs.keys())
            proj_exprs.extend(extra_exprs.values())
    proj = CalciteProjectionNode(proj_cols, proj_exprs)
    self._push(proj)
    self._input_ctx().replace_input_node(frame, proj, proj_cols)
    group = [self._ref_idx(frame, col) for col in op.by]
    fields = op.by.copy()
    aggs = []
    for agg, expr in agg_exprs.items():
        if agg in compound_aggs:
            extra_aggs = compound_aggs[agg].gen_agg_exprs()
            fields.extend(extra_aggs.keys())
            aggs.extend(extra_aggs.values())
        else:
            fields.append(agg)
            aggs.append(self._translate(expr))
    node = CalciteAggregateNode(fields, group, aggs)
    self._push(node)
    if compound_aggs:
        self._input_ctx().replace_input_node(frame, node, fields)
        proj_cols = op.by.copy()
        proj_exprs = [self._ref(frame, col) for col in proj_cols]
        proj_cols.extend(agg_exprs.keys())
        for agg in agg_exprs:
            if agg in compound_aggs:
                proj_exprs.append(compound_aggs[agg].gen_reduce_expr())
            else:
                proj_exprs.append(self._ref(frame, agg))
        proj = CalciteProjectionNode(proj_cols, proj_exprs)
        self._push(proj)
    if op.groupby_opts['sort']:
        collation = [CalciteCollation(col) for col in group]
        self._push(CalciteSortNode(collation))