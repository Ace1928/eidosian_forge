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
@staticmethod
def _before_general_expression(visitor, child):
    return (True, None)