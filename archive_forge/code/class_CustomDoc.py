import re as _re
from .base import build_param_doc as _build_param_doc
class CustomDoc(NDArrayDoc):
    """
    Example
    -------
    Applies a custom operator named `my_custom_operator` to `input`.

    >>> output = mx.symbol.Custom(op_type='my_custom_operator', data=input)
    """