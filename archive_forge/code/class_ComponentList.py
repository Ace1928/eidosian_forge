import logging
from weakref import ref as weakref_ref
from pyomo.common.log import is_debug_set
from pyomo.core.base.set_types import Any
from pyomo.core.base.var import IndexedVar, _VarData
from pyomo.core.base.constraint import IndexedConstraint, _ConstraintData
from pyomo.core.base.objective import IndexedObjective, _ObjectiveData
from pyomo.core.base.expression import IndexedExpression, _ExpressionData
from collections.abc import MutableSequence
class ComponentList(MutableSequence):

    def __init__(self, interface_datatype, *args):
        self._interface_datatype = interface_datatype
        self._data = []
        if len(args) > 0:
            if len(args) > 1:
                raise TypeError('ComponentList expected at most 1 arguments, got %s' % len(args))
            for item in args[0]:
                self.append(item)

    def construct(self, data=None):
        if is_debug_set(logger):
            logger.debug('Constructing ComponentList object, name=%s, from data=%s' % (self.name, str(data)))
        if self._constructed:
            return
        self._constructed = True

    def keys(self):
        return range(len(self))
    iterkeys = keys

    def values(self):
        return list(iter(self))
    itervalues = values

    def items(self):
        return zip(self.keys(), self.values())
    iteritems = items

    def __setitem__(self, i, item):
        if isinstance(item, self._interface_datatype):
            if item._component is None:
                item._component = weakref_ref(self)
                if hasattr(self, '_active'):
                    self._active |= getattr(item, '_active', True)
                self._data[i]._component = None
                self._data[i] = item
                item._index = i
                return
            elif self._data[i] is item:
                return
            raise ValueError('Invalid component object assignment to ComponentList %s at index %s. A parent component has already been assigned the object: %s' % (self.name, i, item.parent_component().name))
        raise TypeError('ComponentList must be assigned objects of type %s. Invalid type for key %s: %s' % (self._interface_datatype.__name__, i, type(item)))

    def insert(self, i, item):
        if isinstance(item, self._interface_datatype):
            if item._component is None:
                item._component = weakref_ref(self)
                if hasattr(self, '_active'):
                    self._active |= getattr(item, '_active', True)
                self._data.insert(i, item)
                item._index = i
                return
            raise ValueError('Invalid component object assignment to ComponentList %s at index %s. A parent component has already been assigned the object: %s' % (self.name, i, item.parent_component().name))
        raise TypeError('ComponentList must be assigned objects of type %s. Invalid type for key %s: %s' % (self._interface_datatype.__name__, i, type(item)))

    def __delitem__(self, i):
        obj = self._data[i]
        obj._component = None
        del self._data[i]

    def __getitem__(self, i):
        return self._data[i]

    def __len__(self):
        return self._data.__len__()

    def __contains__(self, item):
        item_id = id(item)
        return any((item_id == id(_v) for _v in self._data))

    def index(self, item, start=0, stop=None):
        """S.index(value, [start, [stop]]) -> integer -- return first index of value.

        Raises ValueError if the value is not present.
        """
        if start is not None and start < 0:
            start = max(len(self) + start, 0)
        if stop is not None and stop < 0:
            stop += len(self)
        item_id = id(item)
        i = start
        while stop is None or i < stop:
            try:
                if id(self[i]) == item_id:
                    return i
            except IndexError:
                break
            i += 1
        raise ValueError

    def count(self, item):
        """S.count(value) -> integer -- return number of occurrences of value"""
        item_id = id(item)
        cnt = sum((1 for _v in self._data if id(_v) == item_id))
        assert cnt == 1
        return cnt

    def reverse(self):
        """S.reverse() -- reverse *IN PLACE*"""
        n = len(self)
        data = self._data
        for i in range(n // 2):
            data[i], data[n - i - 1] = (data[n - i - 1], data[i])