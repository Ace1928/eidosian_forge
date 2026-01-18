from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Type, Union
from antlr4.ParserRuleContext import ParserRuleContext
from triad import assert_or_throw
from _qpd_antlr import QPDParser as qp
from _qpd_antlr import QPDVisitor
from qpd.dataframe import Column, DataFrame, DataFrames
from qpd._parser.utils import (
from qpd.specs import (
from qpd.workflow import WorkflowDataFrame
class ColumnMentionVisitor(VisitorBase):

    def __init__(self, context: VisitorContext):
        super().__init__(context)
        self._mentions = ColumnMentions()

    def get_mentions(self, ctx: Any) -> ColumnMentions:
        self._mentions = ColumnMentions()
        if ctx is not None:
            self.visit(ctx)
        return self._mentions

    def visitStar(self, ctx: qp.StarContext) -> None:
        self.not_support(ctx)

    def visitColumnReference(self, ctx: qp.ColumnReferenceContext) -> None:
        self._mentions.add(self.add_column_mentions(ctx))

    def visitDereference(self, ctx: qp.DereferenceContext) -> None:
        self._mentions.add(self.add_column_mentions(ctx))