from abc import ABCMeta, abstractmethod, abstractproperty
from typing import Dict as ptDict, Type as ptType
import itertools
import weakref
from functools import cached_property
import numpy as np
from numba.core.utils import get_hashable_key
class TypeRef(Dummy):
    """Reference to a type.

    Used when a type is passed as a value.
    """

    def __init__(self, instance_type):
        self.instance_type = instance_type
        super(TypeRef, self).__init__('typeref[{}]'.format(self.instance_type))

    @property
    def key(self):
        return self.instance_type