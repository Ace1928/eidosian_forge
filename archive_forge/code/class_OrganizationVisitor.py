from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Type, Union
from antlr4.ParserRuleContext import ParserRuleContext
from triad import assert_or_throw
from _qpd_antlr import QPDParser as qp
from _qpd_antlr import QPDVisitor
from qpd.dataframe import Column, DataFrame, DataFrames
from qpd._parser.utils import (
from qpd.specs import (
from qpd.workflow import WorkflowDataFrame
class OrganizationVisitor(VisitorBase):

    def __init__(self, context: VisitorContext):
        super().__init__(context)

    def visitSortItem(self, ctx: qp.SortItemContext) -> OrderItemSpec:
        name = self.to_str(ctx.expression())
        asc = ctx.DESC() is None
        if ctx.FIRST() is None and ctx.LAST() is None:
            na_position = 'auto'
        else:
            na_position = 'first' if ctx.FIRST() is not None else 'last'
        return OrderItemSpec(name, asc, na_position)

    def visitQueryOrganization(self, ctx: qp.QueryOrganizationContext) -> Tuple[OrderBySpec, int]:
        self.assert_none(ctx.clusterBy, ctx.distributeBy, ctx.sort, ctx.windowClause())
        if ctx.order is not None:
            items = [self.visitSortItem(x) for x in ctx.order]
            order_by = OrderBySpec(*items)
        else:
            order_by = OrderBySpec()
        if ctx.limit is not None:
            is_col, _ = PreFunctionArgument(self).visit_argument(ctx.limit)
            self.assert_support(not is_col, ctx)
            limit = int(eval(self.to_str(ctx.limit)))
            self.assert_support(limit >= 0, ctx)
        else:
            limit = -1
        return (order_by, limit)

    def organize(self, df: DataFrame, ctx: Optional[qp.QueryOrganizationContext]) -> DataFrame:
        if ctx is None:
            return df
        order_by, limit = self.visitQueryOrganization(ctx)
        if len(order_by) == 0 and limit < 0:
            return df
        return self.workflow.op_to_df(list(df.keys()), 'order_by_limit', df, order_by, limit)