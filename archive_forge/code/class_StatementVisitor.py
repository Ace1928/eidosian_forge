from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Type, Union
from antlr4.ParserRuleContext import ParserRuleContext
from triad import assert_or_throw
from _qpd_antlr import QPDParser as qp
from _qpd_antlr import QPDVisitor
from qpd.dataframe import Column, DataFrame, DataFrames
from qpd._parser.utils import (
from qpd.specs import (
from qpd.workflow import WorkflowDataFrame
class StatementVisitor(VisitorBase):

    def __init__(self, context: VisitorContext):
        super().__init__(context)

    def visitRegularQuerySpecification(self, ctx: qp.RegularQuerySpecificationContext) -> DataFrame:
        self.copy(PreScan).visit(ctx)
        self.copy(FromVisitor).visit(ctx)
        if ctx.whereClause() is not None:
            self.copy(WhereVisitor).visit(ctx.whereClause())
        if ctx.aggregationClause() is not None or self.get('has_agg_func', False):
            self.copy(AggregationVisitor).visit(ctx)
        elif self.get('has_window_func', False):
            self.copy(WindowVisitor).visit(ctx)
        if ctx.havingClause() is not None:
            self.copy(WhereVisitor).visit(ctx.havingClause())
        self.copy(SelectVisitor).visit(ctx.selectClause())
        return self.current

    def visitSetOperation(self, ctx: qp.SetOperationContext) -> DataFrame:
        op = self.to_str(ctx.operator).lower()
        self.assert_support(op in ['union', 'intersect', 'except'], op)
        if op == 'except':
            op = 'except_df'
        unique = ctx.setQuantifier() is None or ctx.setQuantifier().DISTINCT() is not None
        v = StatementVisitor(VisitorContext(sql=self.sql, workflow=self.workflow, dfs=DataFrames(self.dfs)))
        left = v.visit(ctx.left)
        v = StatementVisitor(VisitorContext(sql=self.sql, workflow=self.workflow, dfs=DataFrames(self.dfs)))
        right = v.visit(ctx.right)
        return self.workflow.op_to_df(list(left.keys()), op, left, right, unique=unique)

    def visitQuery(self, ctx: qp.QueryContext) -> DataFrame:
        if ctx.ctes() is not None:
            self.visit(ctx.ctes())
        df = self.visit(ctx.queryTerm())
        return OrganizationVisitor(self).organize(df, ctx.queryOrganization())

    def visitSingleStatement(self, ctx: qp.SingleStatementContext):
        return self.visit(ctx.statement())

    def visitCtes(self, ctx: qp.CtesContext) -> None:
        for c in ctx.namedQuery():
            name, df = self.visitNamedQuery(c)
            self.set('dfs', DataFrames(self.dfs, {name: df}))

    def visitNamedQuery(self, ctx: qp.NamedQueryContext) -> Tuple[str, DataFrame]:
        self.assert_none(ctx.identifierList())
        name = self.to_str(ctx.name, '')
        v = StatementVisitor(VisitorContext(sql=self.sql, workflow=self.workflow, dfs=DataFrames(self.dfs)))
        df = v.visit(ctx.query())
        return (name, df)