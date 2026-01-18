import builtins
import itertools
import logging
import math
import operator
import sys
from functools import lru_cache
from typing import Optional, Type, TYPE_CHECKING, Union
from torch import (  # noqa: F401
from torch.fx.experimental._sym_dispatch_mode import (
def _make_user_magic(method, user_type):
    if method in magic_methods_on_operator_with_trailing_underscore:
        method_attr = f'{method}_'
    else:
        method_attr = method

    def get_constant(x: Union[SymInt, int, SymFloat, float, SymBool, bool]):
        if isinstance(x, (int, float, bool)):
            return x
        if isinstance(x, SymBool):
            return x.node.guard_bool('', 0)
        raise AssertionError('expect to be called with constant SymBools')

    def is_constant(x):
        if isinstance(x, (int, float, bool)):
            return True
        if isinstance(x, (SymInt, SymFloat, SymBool)):
            return x.node.is_constant()
        return False
    if method in bool_becomes_int_magic_methods:

        def promote(x):
            """Implements True+True=2, which works in python but not sympy"""
            if isinstance(x, SymBool):
                return SymInt(x.node.wrap_int(int(x)))
            return x
    else:

        def promote(x):
            return x

    def unary_magic_impl(self):
        self = promote(self)
        if is_constant(self):
            return method_to_operator(method)(get_constant(self))
        return wrap_node(getattr(self.node, method_attr)())

    def binary_magic_impl(self, other):
        self = promote(self)
        other = promote(other)
        if is_constant(self):
            return method_to_operator(method)(get_constant(self), other)
        if is_constant(other):
            other = get_constant(other)
        other_node = to_node(self.node, other)
        if other_node is NotImplemented:
            return NotImplemented
        ret = wrap_node(getattr(self.node, method_attr)(other_node))
        return get_constant(ret) if is_constant(ret) else ret

    def rbinary_magic_impl(self, other):
        self = promote(self)
        other = promote(other)
        if is_constant(self):
            return method_to_operator(method)(get_constant(self), other)
        if is_constant(other):
            other = get_constant(other)
        other_node = to_node(self.node, other)
        if other_node is NotImplemented:
            return NotImplemented
        ret = wrap_node(getattr(other_node, method_attr)(self.node))
        return get_constant(ret) if is_constant(ret) else ret
    if method in unary_magic_methods:
        setattr(user_type, f'__{method}__', unary_magic_impl)
    elif method == 'sym_ite':

        def sym_ite_magic_impl(pred, then_val, else_val):
            pred_node = pred.node
            then_node = to_node(pred_node, then_val)
            else_node = to_node(pred_node, else_val)
            if then_node is NotImplemented or else_node is NotImplemented:
                return NotImplemented
            assert isinstance(then_node, SymNode) and isinstance(else_node, SymNode) and (then_node.pytype == else_node.pytype)
            ret = wrap_node(getattr(pred.node, method_attr)(then_node, else_node))
            return get_constant(ret) if ret.node.is_constant() else ret
        setattr(user_type, f'__{method}__', sym_ite_magic_impl)
    else:
        setattr(user_type, f'__{method}__', binary_magic_impl)
        if method in reflectable_magic_methods:
            setattr(user_type, f'__r{method}__', rbinary_magic_impl)