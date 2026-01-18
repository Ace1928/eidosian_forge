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
def _rehydrate_skeleton_class(skeleton_class, class_dict):
    """Put attributes from `class_dict` back on `skeleton_class`.

    See CloudPickler.save_dynamic_class for more info.
    """
    registry = None
    for attrname, attr in class_dict.items():
        if attrname == '_abc_impl':
            registry = attr
        else:
            setattr(skeleton_class, attrname, attr)
    if registry is not None:
        for subclass in registry:
            skeleton_class.register(subclass)
    return skeleton_class