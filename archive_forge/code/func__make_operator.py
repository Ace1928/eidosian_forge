import functools
import types
from ._make import _make_ne
def _make_operator(name, func):
    """
    Create operator method.
    """

    def method(self, other):
        if not self._is_comparable_to(other):
            return NotImplemented
        result = func(self.value, other.value)
        if result is NotImplemented:
            return NotImplemented
        return result
    method.__name__ = f'__{name}__'
    method.__doc__ = f'Return a {_operation_names[name]} b.  Computed by attrs.'
    return method