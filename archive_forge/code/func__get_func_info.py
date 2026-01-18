from importlib import import_module
from inspect import signature
from numbers import Integral, Real
import pytest
from sklearn.utils._param_validation import (
def _get_func_info(func_module):
    module_name, func_name = func_module.rsplit('.', 1)
    module = import_module(module_name)
    func = getattr(module, func_name)
    func_sig = signature(func)
    func_params = [p.name for p in func_sig.parameters.values() if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)]
    required_params = [p.name for p in func_sig.parameters.values() if p.default is p.empty and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)]
    return (func, func_name, func_params, required_params)