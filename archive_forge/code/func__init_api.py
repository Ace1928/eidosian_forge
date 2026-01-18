import os
import sys
import ctypes
from ..base import _LIB, check_call
from .base import py_str, c_str
def _init_api(namespace, target_module_name=None):
    """Initialize api for a given module name

    namespace : str
       The namespace of the source registry

    target_module_name : str
       The target module name if different from namespace
    """
    target_module_name = target_module_name if target_module_name else namespace
    if namespace.startswith('mxnet.'):
        _init_api_prefix(target_module_name, namespace[6:])
    else:
        _init_api_prefix(target_module_name, namespace)