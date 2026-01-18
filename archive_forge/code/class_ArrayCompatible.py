from abc import ABCMeta, abstractmethod, abstractproperty
from typing import Dict as ptDict, Type as ptType
import itertools
import weakref
from functools import cached_property
import numpy as np
from numba.core.utils import get_hashable_key
class ArrayCompatible(Type):
    """
    Type class for Numpy array-compatible objects (typically, objects
    exposing an __array__ method).
    Derived classes should implement the *as_array* attribute.
    """
    array_priority = 0.0

    @abstractproperty
    def as_array(self):
        """
        The equivalent array type, for operations supporting array-compatible
        objects (such as ufuncs).
        """

    @cached_property
    def ndim(self):
        return self.as_array.ndim

    @cached_property
    def layout(self):
        return self.as_array.layout

    @cached_property
    def dtype(self):
        return self.as_array.dtype