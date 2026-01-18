from __future__ import absolute_import
from functools import partial
import inspect
import pprint
import sys
from types import ModuleType
import six
from six import wraps
import mock
def _set_return_value(mock, method, name):
    fixed = _return_values.get(name, DEFAULT)
    if fixed is not DEFAULT:
        method.return_value = fixed
        return
    return_calulator = _calculate_return_value.get(name)
    if return_calulator is not None:
        try:
            return_value = return_calulator(mock)
        except AttributeError:
            return_value = AttributeError(name)
        method.return_value = return_value
        return
    side_effector = _side_effect_methods.get(name)
    if side_effector is not None:
        method.side_effect = side_effector(mock)