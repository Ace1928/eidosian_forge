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
def _get_or_create_tracker_id(class_def):
    with _DYNAMIC_CLASS_TRACKER_LOCK:
        class_tracker_id = _DYNAMIC_CLASS_TRACKER_BY_CLASS.get(class_def)
        if class_tracker_id is None:
            class_tracker_id = uuid.uuid4().hex
            _DYNAMIC_CLASS_TRACKER_BY_CLASS[class_def] = class_tracker_id
            _DYNAMIC_CLASS_TRACKER_BY_ID[class_tracker_id] = class_def
    return class_tracker_id