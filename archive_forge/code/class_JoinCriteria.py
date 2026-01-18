from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Type, Union
from antlr4.ParserRuleContext import ParserRuleContext
from triad import assert_or_throw
from _qpd_antlr import QPDParser as qp
from _qpd_antlr import QPDVisitor
from qpd.dataframe import Column, DataFrame, DataFrames
from qpd._parser.utils import (
from qpd.specs import (
from qpd.workflow import WorkflowDataFrame
class JoinCriteria(VisitorBase):

    def __init__(self, context: VisitorContext):
        super().__init__(context)

    def visitJoinCriteria(self, ctx: qp.JoinCriteriaContext) -> List[List[qp.ValueExpressionContext]]:
        self.assert_none(ctx.identifierList())
        res = self.visit(ctx.booleanExpression())
        self.assert_support(res is not None, ctx)
        return list(res)

    def visitPredicated(self, ctx: qp.PredicatedContext) -> Iterable[List[qp.ValueExpressionContext]]:
        self.assert_none(ctx.predicate())
        res = self.visit(ctx.valueExpression())
        self.assert_support(res is not None, ctx)
        return res

    def visitLogicalBinary(self, ctx: qp.LogicalBinaryContext) -> Iterable[List[qp.ValueExpressionContext]]:
        self.assert_support(ctx.AND() is not None, ctx)
        for x in self.visit(ctx.left):
            yield x
        for y in self.visit(ctx.right):
            yield y

    def visitComparison(self, ctx: qp.ComparisonContext) -> Iterable[List[qp.ValueExpressionContext]]:
        self.assert_support(ctx.comparisonOperator().EQ() is not None, ctx)
        v = self.copy(ColumnMentionVisitor)
        left = v.get_mentions(ctx.left)
        left_ctx = ctx.left
        v = self.copy(ColumnMentionVisitor)
        right = v.get_mentions(ctx.right)
        right_ctx = ctx.right
        yield self._fit(left, left_ctx, right, right_ctx)

    def _fit(self, left_mentions: ColumnMentions, left_ctx: qp.ValueExpressionContext, right_mentions: ColumnMentions, right_ctx: qp.ValueExpressionContext) -> List[qp.ValueExpressionContext]:
        left_set = set((x.df_name for x in left_mentions))
        right_set = set((x.df_name for x in right_mentions))
        joined_set = self._joined
        current_set = set([self._right_name])
        if left_set == current_set and right_set.issubset(joined_set):
            return [right_ctx, left_ctx]
        if right_set == current_set and left_set.issubset(joined_set):
            return [left_ctx, right_ctx]
        raise NotImplementedError(self.to_str(left_ctx), self.to_str(right_ctx))