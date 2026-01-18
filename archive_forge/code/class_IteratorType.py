from abc import ABCMeta, abstractmethod, abstractproperty
from typing import Dict as ptDict, Type as ptType
import itertools
import weakref
from functools import cached_property
import numpy as np
from numba.core.utils import get_hashable_key
class IteratorType(IterableType):
    """
    Base class for all iterator types.
    Derived classes should implement the *yield_type* attribute.
    """

    def __init__(self, name, **kwargs):
        super(IteratorType, self).__init__(name, **kwargs)

    @abstractproperty
    def yield_type(self):
        """
        The type of values yielded by the iterator.
        """

    @property
    def iterator_type(self):
        return self