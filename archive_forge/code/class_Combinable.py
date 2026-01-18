import copy
import datetime
import functools
import inspect
from collections import defaultdict
from decimal import Decimal
from types import NoneType
from uuid import UUID
from django.core.exceptions import EmptyResultSet, FieldError, FullResultSet
from django.db import DatabaseError, NotSupportedError, connection
from django.db.models import fields
from django.db.models.constants import LOOKUP_SEP
from django.db.models.query_utils import Q
from django.utils.deconstruct import deconstructible
from django.utils.functional import cached_property
from django.utils.hashable import make_hashable
class Combinable:
    """
    Provide the ability to combine one or two objects with
    some connector. For example F('foo') + F('bar').
    """
    ADD = '+'
    SUB = '-'
    MUL = '*'
    DIV = '/'
    POW = '^'
    MOD = '%%'
    BITAND = '&'
    BITOR = '|'
    BITLEFTSHIFT = '<<'
    BITRIGHTSHIFT = '>>'
    BITXOR = '#'

    def _combine(self, other, connector, reversed):
        if not hasattr(other, 'resolve_expression'):
            other = Value(other)
        if reversed:
            return CombinedExpression(other, connector, self)
        return CombinedExpression(self, connector, other)

    def __neg__(self):
        return self._combine(-1, self.MUL, False)

    def __add__(self, other):
        return self._combine(other, self.ADD, False)

    def __sub__(self, other):
        return self._combine(other, self.SUB, False)

    def __mul__(self, other):
        return self._combine(other, self.MUL, False)

    def __truediv__(self, other):
        return self._combine(other, self.DIV, False)

    def __mod__(self, other):
        return self._combine(other, self.MOD, False)

    def __pow__(self, other):
        return self._combine(other, self.POW, False)

    def __and__(self, other):
        if getattr(self, 'conditional', False) and getattr(other, 'conditional', False):
            return Q(self) & Q(other)
        raise NotImplementedError('Use .bitand(), .bitor(), and .bitxor() for bitwise logical operations.')

    def bitand(self, other):
        return self._combine(other, self.BITAND, False)

    def bitleftshift(self, other):
        return self._combine(other, self.BITLEFTSHIFT, False)

    def bitrightshift(self, other):
        return self._combine(other, self.BITRIGHTSHIFT, False)

    def __xor__(self, other):
        if getattr(self, 'conditional', False) and getattr(other, 'conditional', False):
            return Q(self) ^ Q(other)
        raise NotImplementedError('Use .bitand(), .bitor(), and .bitxor() for bitwise logical operations.')

    def bitxor(self, other):
        return self._combine(other, self.BITXOR, False)

    def __or__(self, other):
        if getattr(self, 'conditional', False) and getattr(other, 'conditional', False):
            return Q(self) | Q(other)
        raise NotImplementedError('Use .bitand(), .bitor(), and .bitxor() for bitwise logical operations.')

    def bitor(self, other):
        return self._combine(other, self.BITOR, False)

    def __radd__(self, other):
        return self._combine(other, self.ADD, True)

    def __rsub__(self, other):
        return self._combine(other, self.SUB, True)

    def __rmul__(self, other):
        return self._combine(other, self.MUL, True)

    def __rtruediv__(self, other):
        return self._combine(other, self.DIV, True)

    def __rmod__(self, other):
        return self._combine(other, self.MOD, True)

    def __rpow__(self, other):
        return self._combine(other, self.POW, True)

    def __rand__(self, other):
        raise NotImplementedError('Use .bitand(), .bitor(), and .bitxor() for bitwise logical operations.')

    def __ror__(self, other):
        raise NotImplementedError('Use .bitand(), .bitor(), and .bitxor() for bitwise logical operations.')

    def __rxor__(self, other):
        raise NotImplementedError('Use .bitand(), .bitor(), and .bitxor() for bitwise logical operations.')

    def __invert__(self):
        return NegatedExpression(self)