from abc import ABCMeta, abstractmethod, abstractproperty
from typing import Dict as ptDict, Type as ptType
import itertools
import weakref
from functools import cached_property
import numpy as np
from numba.core.utils import get_hashable_key
class IterableType(Type):
    """
    Base class for iterable types.
    """

    @abstractproperty
    def iterator_type(self):
        """
        The iterator type obtained when calling iter() (explicitly or implicitly).
        """