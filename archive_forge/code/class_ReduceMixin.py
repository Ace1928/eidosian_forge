import sys
import abc
import io
import copyreg
import pickle
from numba import cloudpickle
from llvmlite import ir
class ReduceMixin(abc.ABC):
    """A mixin class for objects that should be reduced by the NumbaPickler
    instead of the standard pickler.
    """

    @abc.abstractmethod
    def _reduce_states(self):
        raise NotImplementedError

    @abc.abstractclassmethod
    def _rebuild(cls, **kwargs):
        raise NotImplementedError

    def _reduce_class(self):
        return self.__class__

    def __reduce__(self):
        return custom_reduce(self._reduce_class(), self._reduce_states())