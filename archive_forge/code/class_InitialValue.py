from abc import ABCMeta, abstractmethod, abstractproperty
from typing import Dict as ptDict, Type as ptType
import itertools
import weakref
from functools import cached_property
import numpy as np
from numba.core.utils import get_hashable_key
class InitialValue(object):
    """
    Used as a mixin for a type will potentially have an initial value that will
    be carried in the .initial_value attribute.
    """

    def __init__(self, initial_value):
        self._initial_value = initial_value

    @property
    def initial_value(self):
        return self._initial_value