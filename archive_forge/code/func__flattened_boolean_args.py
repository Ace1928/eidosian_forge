import types
from itertools import islice
import logging
import traceback
from pyomo.common.errors import PyomoException, DeveloperError
from pyomo.common.deprecation import (
from .numvalue import (
from .base import ExpressionBase
from .boolean_value import BooleanValue, BooleanConstant
from .expr_common import _and, _or, _equiv, _inv, _xor, _impl, ExpressionType
from .numeric_expr import NumericExpression
import operator
def _flattened_boolean_args(args):
    """Flatten any potentially indexed arguments and check that they are
    Boolean-valued."""
    for arg in args:
        if arg.__class__ in native_types:
            myiter = (arg,)
        elif isinstance(arg, (types.GeneratorType, list)):
            myiter = arg
        elif arg.is_indexed():
            myiter = arg.values()
        else:
            myiter = (arg,)
        for _argdata in myiter:
            if _argdata.__class__ in native_logical_types:
                yield _argdata
            elif hasattr(_argdata, 'is_logical_type') and _argdata.is_logical_type():
                yield _argdata
            elif isinstance(_argdata, BooleanValue):
                yield _argdata
            else:
                raise ValueError("Non-Boolean-valued argument '%s' encountered when constructing expression of Boolean arguments" % arg)