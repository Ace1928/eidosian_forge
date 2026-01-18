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
def _extract_class_dict(cls):
    """Retrieve a copy of the dict of a class without the inherited methods"""
    clsdict = dict(cls.__dict__)
    if len(cls.__bases__) == 1:
        inherited_dict = cls.__bases__[0].__dict__
    else:
        inherited_dict = {}
        for base in reversed(cls.__bases__):
            inherited_dict.update(base.__dict__)
    to_remove = []
    for name, value in clsdict.items():
        try:
            base_value = inherited_dict[name]
            if value is base_value:
                to_remove.append(name)
        except KeyError:
            pass
    for name in to_remove:
        clsdict.pop(name)
    return clsdict