import numpy
from modin.error_message import ErrorMessage
from modin.utils import _inherit_docstrings
from .arr import array
from .utils import try_convert_from_interoperable_type
def _dispatch_math(operator_name, arr_method_name=None):

    @_inherit_docstrings(getattr(numpy, operator_name))
    def call(x, *args, **kwargs):
        x = try_convert_from_interoperable_type(x)
        if not isinstance(x, array):
            ErrorMessage.bad_type_for_numpy_op(operator_name, type(x))
            return getattr(numpy, operator_name)(x, *args, **kwargs)
        return getattr(x, arr_method_name or operator_name)(*args, **kwargs)
    return call