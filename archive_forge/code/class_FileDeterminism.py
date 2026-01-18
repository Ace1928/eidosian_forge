import collections
import enum
import functools
import itertools
import logging
import operator
import sys
from pyomo.common.collections import Sequence, ComponentMap, ComponentSet
from pyomo.common.deprecation import deprecation_warning
from pyomo.common.errors import DeveloperError, InvalidValueError
from pyomo.common.numeric_types import (
from pyomo.core.pyomoobject import PyomoObject
from pyomo.core.base import (
from pyomo.core.base.component import ActiveComponent
from pyomo.core.base.expression import _ExpressionData
from pyomo.core.expr.numvalue import is_fixed, value
import pyomo.core.expr as EXPR
import pyomo.core.kernel as kernel
class FileDeterminism(enum.IntEnum):
    NONE = 0
    ORDERED = 10
    SORT_INDICES = 20
    SORT_SYMBOLS = 30

    def __str__(self):
        return enum.Enum.__str__(self)

    def __format__(self, spec):
        return enum.Enum.__format__(self, spec)

    @classmethod
    def _missing_(cls, value):
        if value in _FileDeterminism_deprecation:
            new = FileDeterminism(_FileDeterminism_deprecation[value])
            deprecation_warning(f'FileDeterminism({value}) is deprecated.  Please use {str(new)} ({int(new)})', version='6.5.0')
            return new
        return super()._missing_(value)