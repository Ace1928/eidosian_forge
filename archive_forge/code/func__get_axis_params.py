import numpy as np
from functools import wraps
from scipy._lib._docscrape import FunctionDoc, Parameter
from scipy._lib._util import _contains_nan, AxisError, _get_nan
import inspect
def _get_axis_params(default_axis=0, _name=_name, _desc=_desc):
    _type = f'int or None, default: {default_axis}'
    _axis_parameter_doc = Parameter(_name, _type, _desc)
    _axis_parameter = inspect.Parameter(_name, inspect.Parameter.KEYWORD_ONLY, default=default_axis)
    return (_axis_parameter_doc, _axis_parameter)