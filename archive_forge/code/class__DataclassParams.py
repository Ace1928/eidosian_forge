import re
import sys
import copy
import types
import inspect
import keyword
import builtins
import functools
import itertools
import abc
import _thread
from types import FunctionType, GenericAlias
class _DataclassParams:
    __slots__ = ('init', 'repr', 'eq', 'order', 'unsafe_hash', 'frozen')

    def __init__(self, init, repr, eq, order, unsafe_hash, frozen):
        self.init = init
        self.repr = repr
        self.eq = eq
        self.order = order
        self.unsafe_hash = unsafe_hash
        self.frozen = frozen

    def __repr__(self):
        return f'_DataclassParams(init={self.init!r},repr={self.repr!r},eq={self.eq!r},order={self.order!r},unsafe_hash={self.unsafe_hash!r},frozen={self.frozen!r})'