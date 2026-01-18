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
def _lookup_class_or_track(class_tracker_id, class_def):
    if class_tracker_id is not None:
        with _DYNAMIC_CLASS_TRACKER_LOCK:
            class_def = _DYNAMIC_CLASS_TRACKER_BY_ID.setdefault(class_tracker_id, class_def)
            _DYNAMIC_CLASS_TRACKER_BY_CLASS[class_def] = class_tracker_id
    return class_def