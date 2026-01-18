import logging
import sys
from pyomo.common.dependencies import numpy_available
from pyomo.common.deprecation import deprecated, relocated_module_attribute
from pyomo.common.errors import TemplateExpressionError
def RegisterComplexType(new_type: type):
    """Register the specified type as an "complex type".

    A utility function for registering new types as "native complex
    types".  Complex types can NOT be leaf nodes in Pyomo numeric
    expressions.  The type should be compatible with :py:class:`complex`
    (that is, store a scalar complex value and be castable to a Python
    complex).

    Note that complex types are NOT registered as logical or numeric types.

    Parameters
    ----------
    new_type: type
        The new complex type (e.g, numpy.complex128)

    """
    native_types.add(new_type)
    native_complex_types.add(new_type)
    nonpyomo_leaf_types.add(new_type)