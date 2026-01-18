from __future__ import annotations
import typing
from . import expr
class _StructuralEquivalenceImpl(ExprVisitor[bool]):
    __slots__ = ('self_key', 'other_key', 'other')

    def __init__(self, other: expr.Expr, self_key, other_key):
        self.self_key = self_key
        self.other_key = other_key
        self.other = other

    def visit_var(self, node, /):
        if self.other.__class__ is not node.__class__ or self.other.type != node.type:
            return False
        if self.self_key is None or (self_var := self.self_key(node.var)) is None:
            self_var = node.var
        if self.other_key is None or (other_var := self.other_key(self.other.var)) is None:
            other_var = self.other.var
        return self_var == other_var

    def visit_value(self, node, /):
        return node.__class__ is self.other.__class__ and node.type == self.other.type and (node.value == self.other.value)

    def visit_unary(self, node, /):
        if self.other.__class__ is not node.__class__ or self.other.op is not node.op or self.other.type != node.type:
            return False
        self.other = self.other.operand
        return node.operand.accept(self)

    def visit_binary(self, node, /):
        if self.other.__class__ is not node.__class__ or self.other.op is not node.op or self.other.type != node.type:
            return False
        other = self.other
        self.other = other.left
        if not node.left.accept(self):
            return False
        self.other = other.right
        return node.right.accept(self)

    def visit_cast(self, node, /):
        if self.other.__class__ is not node.__class__ or self.other.type != node.type:
            return False
        self.other = self.other.operand
        return node.operand.accept(self)