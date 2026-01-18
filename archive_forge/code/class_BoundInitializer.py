import collections
import functools
import inspect
from collections.abc import Sequence
from collections.abc import Mapping
from pyomo.common.dependencies import numpy, numpy_available, pandas, pandas_available
from pyomo.common.modeling import NOTSET
from pyomo.core.pyomoobject import PyomoObject
class BoundInitializer(InitializerBase):
    """Initializer wrapper for processing bounds (mapping scalars to 2-tuples)

    Note that this class is meant to mimic the behavior of
    :py:func:`Initializer` and will return ``None`` if the initializer
    that it is wrapping is ``None``.

    Parameters
    ----------
    arg:

        As with :py:func:`Initializer`, this is the raw argument passed
        to the component constructor.

    obj: :py:class:`Component`

        The component that "owns" the initializer.  This initializer
        will treat sequences as mappings only if the owning component is
        indexed and the sequence passed to the initializer is not of
        length 2

    """
    __slots__ = ('_initializer',)

    def __new__(cls, arg=None, obj=NOTSET):
        if arg is None and obj is not NOTSET:
            return None
        else:
            return super().__new__(cls)

    def __init__(self, arg, obj=NOTSET):
        if obj is NOTSET or obj.is_indexed():
            treat_sequences_as_mappings = not (isinstance(arg, Sequence) and len(arg) == 2 and (not isinstance(arg[0], Sequence)))
        else:
            treat_sequences_as_mappings = False
        self._initializer = Initializer(arg, treat_sequences_as_mappings=treat_sequences_as_mappings)

    def __call__(self, parent, index):
        val = self._initializer(parent, index)
        if _bound_sequence_types[val.__class__]:
            return val
        if _bound_sequence_types[val.__class__] is None:
            _bound_sequence_types[val.__class__] = isinstance(val, Sequence) and (not isinstance(val, str))
            if _bound_sequence_types[val.__class__]:
                return val
        return (val, val)

    def constant(self):
        """Return True if this initializer is constant across all indices"""
        return self._initializer.constant()

    def contains_indices(self):
        """Return True if this initializer contains embedded indices"""
        return self._initializer.contains_indices()

    def indices(self):
        return self._initializer.indices()