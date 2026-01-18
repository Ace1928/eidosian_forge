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
def _make_typevar(name, bound, constraints, covariant, contravariant, class_tracker_id):
    tv = typing.TypeVar(name, *constraints, bound=bound, covariant=covariant, contravariant=contravariant)
    if class_tracker_id is not None:
        return _lookup_class_or_track(class_tracker_id, tv)
    else:
        return tv