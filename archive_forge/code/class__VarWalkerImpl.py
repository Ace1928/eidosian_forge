from __future__ import annotations
import typing
from . import expr
class _VarWalkerImpl(ExprVisitor[typing.Iterable[expr.Var]]):
    __slots__ = ()

    def visit_var(self, node, /):
        yield node

    def visit_value(self, node, /):
        yield from ()

    def visit_unary(self, node, /):
        yield from node.operand.accept(self)

    def visit_binary(self, node, /):
        yield from node.left.accept(self)
        yield from node.right.accept(self)

    def visit_cast(self, node, /):
        yield from node.operand.accept(self)