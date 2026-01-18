from abc import ABCMeta, abstractmethod, abstractproperty
from typing import Dict as ptDict, Type as ptType
import itertools
import weakref
from functools import cached_property
import numpy as np
from numba.core.utils import get_hashable_key
class DTypeSpec(Type):
    """
    Base class for types usable as "dtype" arguments to various Numpy APIs
    (e.g. np.empty()).
    """

    @abstractproperty
    def dtype(self):
        """
        The actual dtype denoted by this dtype spec (a Type instance).
        """