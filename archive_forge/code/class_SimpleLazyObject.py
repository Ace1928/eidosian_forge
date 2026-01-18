import copy
import itertools
import operator
from functools import wraps
class SimpleLazyObject(LazyObject):
    """
    A lazy object initialized from any function.

    Designed for compound objects of unknown type. For builtins or objects of
    known type, use django.utils.functional.lazy.
    """

    def __init__(self, func):
        """
        Pass in a callable that returns the object to be wrapped.

        If copies are made of the resulting SimpleLazyObject, which can happen
        in various circumstances within Django, then you must ensure that the
        callable can be safely run more than once and will return the same
        value.
        """
        self.__dict__['_setupfunc'] = func
        super().__init__()

    def _setup(self):
        self._wrapped = self._setupfunc()

    def __repr__(self):
        if self._wrapped is empty:
            repr_attr = self._setupfunc
        else:
            repr_attr = self._wrapped
        return '<%s: %r>' % (type(self).__name__, repr_attr)

    def __copy__(self):
        if self._wrapped is empty:
            return SimpleLazyObject(self._setupfunc)
        else:
            return copy.copy(self._wrapped)

    def __deepcopy__(self, memo):
        if self._wrapped is empty:
            result = SimpleLazyObject(self._setupfunc)
            memo[id(self)] = result
            return result
        return copy.deepcopy(self._wrapped, memo)
    __add__ = new_method_proxy(operator.add)

    @new_method_proxy
    def __radd__(self, other):
        return other + self