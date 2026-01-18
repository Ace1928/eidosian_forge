from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Type, Union
from antlr4.ParserRuleContext import ParserRuleContext
from triad import assert_or_throw
from _qpd_antlr import QPDParser as qp
from _qpd_antlr import QPDVisitor
from qpd.dataframe import Column, DataFrame, DataFrames
from qpd._parser.utils import (
from qpd.specs import (
from qpd.workflow import WorkflowDataFrame
class RelationVisitor(VisitorBase):

    def __init__(self, context: VisitorContext):
        super().__init__(context)

    def visitTableAlias(self, ctx: qp.TableAliasContext) -> str:
        self.assert_none(ctx.identifierList())
        return self.to_str(ctx.strictIdentifier(), '')

    def visitTableName(self, ctx: qp.TableNameContext) -> Tuple[DataFrame, str]:
        self.assert_none(ctx.sample())
        df_name = self.to_str(ctx.multipartIdentifier(), '')
        assert_or_throw(df_name in self.dfs, KeyError(f'{df_name} is not found in {list(self.dfs.keys())}'))
        alias = self.visitTableAlias(ctx.tableAlias())
        if alias == '':
            alias = df_name
        return (self.dfs[df_name], alias)

    def visitAliasedQuery(self, ctx: qp.AliasedQueryContext) -> Tuple[DataFrame, str]:
        if ctx.tableAlias().strictIdentifier() is not None:
            name = self.to_str(ctx.tableAlias().strictIdentifier(), '')
        else:
            name = self._obj_to_col_name(self.to_str(ctx.query()))
        v = StatementVisitor(VisitorContext(sql=self.sql, workflow=self.workflow, dfs=DataFrames(self.dfs)))
        df = v.visitQuery(ctx.query())
        return (df, name)

    def visitAliasedRelation(self, ctx: qp.AliasedRelationContext) -> Tuple[DataFrame, str]:
        self.assert_none(ctx.sample())
        df, a = self.copy(RelationVisitor).visit(ctx.relation())
        alias = self.visitTableAlias(ctx.tableAlias())
        return (df, alias if alias != '' else a)

    def visitRelation(self, ctx: qp.RelationContext) -> Tuple[DataFrame, str]:
        self.assert_none(ctx.joinRelation())
        res = self.visit(ctx.relationPrimary())
        if res is None:
            self.not_support(f'{self.to_str(ctx.relationPrimary())} is not supported')
        return res