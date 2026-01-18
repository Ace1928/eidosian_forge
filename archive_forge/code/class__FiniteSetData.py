import inspect
import itertools
import logging
import math
import sys
import weakref
from pyomo.common.pyomo_typing import overload
from pyomo.common.collections import ComponentSet
from pyomo.common.deprecation import deprecated, deprecation_warning, RenamedClass
from pyomo.common.errors import DeveloperError, PyomoException
from pyomo.common.log import is_debug_set
from pyomo.common.modeling import NOTSET
from pyomo.common.sorting import sorted_robust
from pyomo.common.timing import ConstructionTimer
from pyomo.core.expr.numvalue import (
from pyomo.core.base.disable_methods import disable_methods
from pyomo.core.base.initializer import (
from pyomo.core.base.range import (
from pyomo.core.base.component import (
from pyomo.core.base.indexed_component import (
from pyomo.core.base.global_set import (
from collections.abc import Sequence
from operator import itemgetter
class _FiniteSetData(_FiniteSetMixin, _SetData):
    """A general unordered iterable Set"""
    __slots__ = ('_values', '_domain', '_validate', '_filter', '_dimen')

    def __init__(self, component):
        _SetData.__init__(self, component=component)
        if not hasattr(self, '_values'):
            self._values = set()
        self._domain = Any
        self._validate = None
        self._filter = None
        self._dimen = UnknownSetDimen

    def get(self, value, default=None):
        """
        Return True if the set contains a given value.

        This method will raise TypeError for unhashable types.
        """
        if normalize_index.flatten:
            value = normalize_index(value)
        if value in self._values:
            return value
        return default

    def _iter_impl(self):
        return iter(self._values)

    def __reversed__(self):
        try:
            return reversed(self._values)
        except:
            return reversed(self.data())

    def __len__(self):
        """
        Return the number of elements in the set.
        """
        return len(self._values)

    def __str__(self):
        if self.parent_component()._name is not None:
            return self.name
        if not self.parent_component()._constructed:
            return type(self).__name__
        return '{' + ', '.join((str(_) for _ in self)) + '}'

    @property
    def dimen(self):
        if self._dimen is UnknownSetDimen:
            _comp = self.parent_component()
            if not _comp._constructed and _comp._init_dimen.constant():
                return _comp._init_dimen.val
        return self._dimen

    @property
    def domain(self):
        return self._domain

    @property
    @deprecated("'filter' is no longer a public attribute.", version='5.7')
    def filter(self):
        return self._filter

    def add(self, *values):
        count = 0
        _block = self.parent_block()
        for value in values:
            if normalize_index.flatten:
                _value = normalize_index(value)
                if _value.__class__ is tuple:
                    _d = len(_value)
                else:
                    _d = 1
            else:
                _d = 1
                if isinstance(value, Sequence) and self.dimen != 1:
                    _d = len(value)
                _value = value
            if _value not in self._domain:
                raise ValueError('Cannot add value %s to Set %s.\n\tThe value is not in the domain %s' % (value, self.name, self._domain))
            try:
                if _value in self:
                    logger.warning('Element %s already exists in Set %s; no action taken' % (value, self.name))
                    continue
            except:
                exc = sys.exc_info()
                raise TypeError("Unable to insert '%s' into Set %s:\n\t%s: %s" % (value, self.name, exc[0].__name__, exc[1]))
            if self._filter is not None:
                if not self._filter(_block, _value):
                    continue
            if self._validate is not None:
                try:
                    flag = self._validate(_block, _value)
                except:
                    logger.error("Exception raised while validating element '%s' for Set %s" % (value, self.name))
                    raise
                if not flag:
                    raise ValueError('The value=%s violates the validation rule of Set %s' % (value, self.name))
            if self._dimen is not None:
                if _d != self._dimen:
                    if self._dimen is UnknownSetDimen:
                        self._dimen = _d
                    else:
                        raise ValueError('The value=%s has dimension %s and is not valid for Set %s which has dimen=%s' % (value, _d, self.name, self._dimen))
            self._add_impl(_value)
            count += 1
        return count

    def _add_impl(self, value):
        self._values.add(value)

    def remove(self, val):
        self._values.remove(val)

    def discard(self, val):
        self._values.discard(val)

    def clear(self):
        self._values.clear()

    def set_value(self, val):
        self.clear()
        for x in val:
            self.add(x)

    def update(self, values):
        for v in values:
            if v not in self:
                self.add(v)

    def pop(self):
        return self._values.pop()