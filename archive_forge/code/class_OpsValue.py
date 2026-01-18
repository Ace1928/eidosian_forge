from __future__ import annotations
import itertools
from contextlib import contextmanager
from itertools import chain
from threading import local
from typing import Any, Callable, TYPE_CHECKING, Union
from unittest.mock import patch
import sympy
from torch._inductor.utils import IndentedBuffer
from torch.fx.graph import inplace_methods, magic_methods
from .utils import reduction_num_outputs, sympy_str, sympy_symbol
class OpsValue:
    """The return type of most ops calls.

    This exists so we can overload magic methods, and write mathematical
    expressions much more fluently. So instead of

        ops.add(ops.mul(ops.mul(ops.sub(ops.mul(_Ap2, x), _Ap3), x), x), _1)

    we can write

        (_Ap2 * x - _Ap3) * x * x + _1

    """
    value: Any

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return f'OpsValue({self.value!r})'

    def __add__(self, other):
        return ops.add(self, other)

    def __mul__(self, other):
        return ops.mul(self, other)

    def __sub__(self, other):
        return ops.sub(self, other)

    def __neg__(self):
        return ops.neg(self)

    def __truediv__(self, other):
        return ops.truediv(self, other)

    def __floordiv__(self, other):
        return ops.floordiv(self, other)

    def __mod__(self, other):
        return ops.mod(self, other)

    def __pow__(self, other):
        return ops.pow(self, other)