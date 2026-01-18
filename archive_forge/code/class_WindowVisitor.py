from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Type, Union
from antlr4.ParserRuleContext import ParserRuleContext
from triad import assert_or_throw
from _qpd_antlr import QPDParser as qp
from _qpd_antlr import QPDVisitor
from qpd.dataframe import Column, DataFrame, DataFrames
from qpd._parser.utils import (
from qpd.specs import (
from qpd.workflow import WorkflowDataFrame
class WindowVisitor(VisitorBase):

    def __init__(self, context: VisitorContext):
        super().__init__(context)
        self._internal_exprs: Dict[str, Column] = {}
        self._windows: Dict[str, Tuple[List[str], WindowFrameSpec]] = {}
        self._f_to_name: Dict[int, str] = {}

    def visitSortItem(self, ctx: qp.SortItemContext) -> OrderItemSpec:
        name, _ = self._get_internal_col(ctx.expression())
        asc = ctx.DESC() is None
        if ctx.FIRST() is None and ctx.LAST() is None:
            na_position = 'auto'
        else:
            na_position = 'first' if ctx.FIRST() is not None else 'last'
        return OrderItemSpec(name, asc, na_position)

    def visitFrameBound(self, ctx: qp.FrameBoundContext) -> Tuple[bool, int]:
        unbounded = ctx.UNBOUNDED() is not None
        preceding = ctx.PRECEDING() is not None
        current = ctx.CURRENT() is not None
        if current:
            return (False, 0)
        if unbounded:
            return (True, 0)
        n = int(eval(self.to_str(ctx.expression(), '')))
        return (False, -n if preceding else n)

    def visitWindowFrame(self, ctx: qp.WindowFrameContext) -> WindowFrameSpec:
        if ctx is None:
            return make_windowframe_spec('')
        self.assert_support(ctx.ROWS() is not None and ctx.BETWEEN() is not None, ctx)
        s_unbounded, s_n = self.visitFrameBound(ctx.start)
        e_unbounded, e_n = self.visitFrameBound(ctx.end)
        return make_windowframe_spec('rows', None if s_unbounded else s_n, None if e_unbounded else e_n)

    def visitWindowDef(self, ctx: qp.WindowDefContext) -> WindowSpec:
        wf = self.visitWindowFrame(ctx.windowFrame())
        partition_keys = [self._get_internal_col(p)[0] for p in ctx.partition]
        sort = OrderBySpec(*[self.visitSortItem(s) for s in ctx.sortItem()])
        return WindowSpec('', partition_keys=partition_keys, order_by=sort, windowframe=wf)

    def visitFunctionCall(self, ctx: qp.FunctionCallContext) -> Tuple[WindowFunctionSpec, List[Any]]:
        func_name = self.visitFunctionName(ctx.functionName())
        args = ctx.argument
        unique = ctx.setQuantifier() is not None and ctx.setQuantifier().DISTINCT() is not None
        ws = self.visit(ctx.windowSpec())
        func = WindowFunctionSpec(func_name, unique=unique, dropna=False, window=ws)
        return (func, args)

    def visitRegularQuerySpecification(self, ctx: qp.RegularQuerySpecificationContext) -> None:
        self._handle_window_funcs()
        if len(self._internal_exprs) > 0:
            self.update_current(WorkflowDataFrame(self.current, *list(self._internal_exprs.values())))
        for k, v in self._windows.items():
            self.update_current(self.workflow.op_to_df(list(self.current.keys()) + [k], 'window', self.current, v[1], v[0], k))
        self.set('window_func_to_col', self._f_to_name)

    def _handle_window_funcs(self) -> None:
        for f_ctx in self.get('window_funcs', None):
            func, args = self.visit(f_ctx)
            name = self._obj_to_col_name(self.expr(f_ctx))
            args = self._get_func_args(args)
            self._windows[name] = (args, func)
            self._f_to_name[id(f_ctx)] = name

    def _get_func_args(self, args: Any) -> List[ArgumentSpec]:
        e: List[ArgumentSpec] = []
        for i in range(len(args)):
            is_col, is_single = self.get('func_arg_types')[id(args[i])]
            if is_col:
                x, _ = self._get_internal_col(args[i])
                e.append(ArgumentSpec(True, x))
            else:
                e.append(ArgumentSpec(False, eval(self.to_str(args[i]))))
        return e

    def _get_internal_col(self, ctx: Any) -> Tuple[str, Column]:
        internal_expr = self._obj_to_col_name(self.expr(ctx))
        if internal_expr not in self._internal_exprs:
            col = ExpressionVisitor(self)._get_single_column(ctx)
            self._internal_exprs[internal_expr] = col.rename(internal_expr)
        return (internal_expr, self._internal_exprs[internal_expr])