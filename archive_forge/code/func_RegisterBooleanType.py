import logging
import sys
from pyomo.common.dependencies import numpy_available
from pyomo.common.deprecation import deprecated, relocated_module_attribute
from pyomo.common.errors import TemplateExpressionError
@deprecated('The native_boolean_types set (and hence RegisterBooleanType) is deprecated.  Users likely should use RegisterLogicalType.', version='6.6.0')
def RegisterBooleanType(new_type: type):
    """Register the specified type as a "logical type".

    A utility function for registering new types as "native logical
    types".  Logical types can be leaf nodes in Pyomo logical
    expressions.  The type should be compatible with :py:class:`bool`
    (that is, store a scalar and be castable to a Python bool).

    Note that logical types are NOT registered as numeric types.

    Parameters
    ----------
    new_type: type
        The new logical type (e.g, numpy.bool_)

    """
    _native_boolean_types.add(new_type)
    native_types.add(new_type)
    nonpyomo_leaf_types.add(new_type)