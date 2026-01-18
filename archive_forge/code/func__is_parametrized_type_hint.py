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
def _is_parametrized_type_hint(obj):
    type_module = getattr(type(obj), '__module__', None)
    from_typing_extensions = type_module == 'typing_extensions'
    from_typing = type_module == 'typing'
    is_typing = getattr(obj, '__origin__', None) is not None
    is_literal = getattr(obj, '__values__', None) is not None and from_typing_extensions
    is_final = getattr(obj, '__type__', None) is not None and from_typing_extensions
    is_classvar = getattr(obj, '__type__', None) is not None and from_typing
    is_union = getattr(obj, '__union_params__', None) is not None
    is_tuple = getattr(obj, '__tuple_params__', None) is not None
    is_callable = getattr(obj, '__result__', None) is not None and getattr(obj, '__args__', None) is not None
    return any((is_typing, is_literal, is_final, is_classvar, is_union, is_tuple, is_callable))