from __future__ import annotations
import warnings
from .logger import adapter as logger
from .logger import trace as _trace
import os
import sys
import builtins as __builtin__
from pickle import _Pickler as StockPickler, Unpickler as StockUnpickler
from pickle import GLOBAL, POP
from _thread import LockType
from _thread import RLock as RLockType
from types import CodeType, FunctionType, MethodType, GeneratorType, \
from types import MappingProxyType as DictProxyType, new_class
from pickle import DEFAULT_PROTOCOL, HIGHEST_PROTOCOL, PickleError, PicklingError, UnpicklingError
import __main__ as _main_module
import marshal
import gc
import abc
import dataclasses
from weakref import ReferenceType, ProxyType, CallableProxyType
from collections import OrderedDict
from enum import Enum, EnumMeta
from functools import partial
from operator import itemgetter, attrgetter
import importlib.machinery
from types import GetSetDescriptorType, ClassMethodDescriptorType, \
from io import BytesIO as StringIO
from socket import socket as SocketType
from multiprocessing.reduction import _reduce_socket as reduce_socket
import inspect
import typing
from . import _shims
from ._shims import Reduce, Getattr
def _get_typedict_abc(obj, _dict, attrs, postproc_list):
    if hasattr(abc, '_get_dump'):
        registry, _, _, _ = abc._get_dump(obj)
        register = obj.register
        postproc_list.extend(((register, (reg(),)) for reg in registry))
    elif hasattr(obj, '_abc_registry'):
        registry = obj._abc_registry
        register = obj.register
        postproc_list.extend(((register, (reg,)) for reg in registry))
    else:
        raise PicklingError('Cannot find registry of ABC %s', obj)
    if '_abc_registry' in _dict:
        _dict.pop('_abc_registry', None)
        _dict.pop('_abc_cache', None)
        _dict.pop('_abc_negative_cache', None)
    else:
        _dict.pop('_abc_impl', None)
    return (_dict, attrs)