import sys
import abc
import io
import copyreg
import pickle
from numba import cloudpickle
from llvmlite import ir
class _CustomPickled:
    """A wrapper for objects that must be pickled with `NumbaPickler`.

    Standard `pickle` will pick up the implementation registered via `copyreg`.
    This will spawn a `NumbaPickler` instance to serialize the data.

    `NumbaPickler` overrides the handling of this type so as not to spawn a
    new pickler for the object when it is already being pickled by a
    `NumbaPickler`.
    """
    __slots__ = ('ctor', 'states')

    def __init__(self, ctor, states):
        self.ctor = ctor
        self.states = states

    def _reduce(self):
        return (_CustomPickled._rebuild, (self.ctor, self.states))

    @classmethod
    def _rebuild(cls, ctor, states):
        return cls(ctor, states)