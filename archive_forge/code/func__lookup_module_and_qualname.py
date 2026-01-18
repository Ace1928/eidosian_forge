import builtins
import dis
import opcode
import platform
import sys
import types
import weakref
import uuid
import threading
import typing
import warnings
from .compat import pickle
from collections import OrderedDict
from typing import ClassVar, Generic, Union, Tuple, Callable
from pickle import _getattribute
def _lookup_module_and_qualname(obj, name=None):
    if name is None:
        name = getattr(obj, '__qualname__', None)
    if name is None:
        name = getattr(obj, '__name__', None)
    module_name = _whichmodule(obj, name)
    if module_name is None:
        return None
    if module_name == '__main__':
        return None
    module = sys.modules.get(module_name, None)
    if module is None:
        return None
    try:
        obj2, parent = _getattribute(module, name)
    except AttributeError:
        return None
    if obj2 is not obj:
        return None
    return (module, name)