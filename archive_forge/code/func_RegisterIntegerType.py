import logging
import sys
from pyomo.common.dependencies import numpy_available
from pyomo.common.deprecation import deprecated, relocated_module_attribute
from pyomo.common.errors import TemplateExpressionError
def RegisterIntegerType(new_type: type):
    """Register the specified type as an "integer type".

    A utility function for registering new types as "native integer
    types".  Integer types can be leaf nodes in Pyomo numeric
    expressions.  The type should be compatible with :py:class:`float`
    (that is, store a scalar and be castable to a Python float).

    Registering a type as an integer type implies
    :py:func:`RegisterNumericType`.

    Note that integer types are NOT registered as logical / Boolean types.

    Parameters
    ----------
    new_type: type
        The new integer type (e.g, numpy.int64)

    """
    native_numeric_types.add(new_type)
    native_integer_types.add(new_type)
    native_types.add(new_type)
    nonpyomo_leaf_types.add(new_type)