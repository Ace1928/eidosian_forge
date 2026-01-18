from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Type, Union
from antlr4.ParserRuleContext import ParserRuleContext
from triad import assert_or_throw
from _qpd_antlr import QPDParser as qp
from _qpd_antlr import QPDVisitor
from qpd.dataframe import Column, DataFrame, DataFrames
from qpd._parser.utils import (
from qpd.specs import (
from qpd.workflow import WorkflowDataFrame
class PreFunctionArgument(VisitorBase):

    def __init__(self, context: VisitorContext):
        super().__init__(context)
        self._has_star = False
        self._has_func = False
        self._cols = 0
        self._constants = 0

    def visitStar(self, ctx: qp.StarContext) -> None:
        self._cols += 1
        self._has_star = True

    def visitColumnReference(self, ctx: qp.ColumnReferenceContext) -> None:
        self._cols += 1

    def visitDereference(self, ctx: qp.DereferenceContext) -> None:
        self._cols += 1

    def visitConstantDefault(self, ctx: qp.ConstantDefaultContext) -> None:
        self._constants += 1

    def visitFunctionCall(self, ctx: qp.FunctionCallContext) -> None:
        self._has_func = True
        super().visitFunctionCall(ctx)

    def visitLogicalNot(self, ctx: qp.LogicalNotContext) -> None:
        self._has_func = True
        super().visitLogicalNot(ctx)

    def visit_argument(self, ctx: Any) -> Tuple[bool, bool]:
        self._has_star = False
        self._has_func = False
        self._cols = 0
        self._constants = 0
        self.visit(ctx)
        is_col = self._cols > 0
        self.assert_support(is_col or self._constants > 0, ctx)
        if is_col:
            is_single = self._cols == 1 and (not self._has_star) and (not self._has_func) and (self._constants == 0)
        else:
            is_single = self._constants == 1 and (not self._has_func)
        return (is_col, is_single)