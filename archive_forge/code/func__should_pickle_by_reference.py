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
def _should_pickle_by_reference(obj, name=None):
    """Test whether an function or a class should be pickled by reference

    Pickling by reference means by that the object (typically a function or a
    class) is an attribute of a module that is assumed to be importable in the
    target Python environment. Loading will therefore rely on importing the
    module and then calling `getattr` on it to access the function or class.

    Pickling by reference is the only option to pickle functions and classes
    in the standard library. In cloudpickle the alternative option is to
    pickle by value (for instance for interactively or locally defined
    functions and classes or for attributes of modules that have been
    explicitly registered to be pickled by value.
    """
    if isinstance(obj, types.FunctionType) or issubclass(type(obj), type):
        module_and_name = _lookup_module_and_qualname(obj, name=name)
        if module_and_name is None:
            return False
        module, name = module_and_name
        return not _is_registered_pickle_by_value(module)
    elif isinstance(obj, types.ModuleType):
        if _is_registered_pickle_by_value(obj):
            return False
        return obj.__name__ in sys.modules
    else:
        raise TypeError('cannot check importability of {} instances'.format(type(obj).__name__))