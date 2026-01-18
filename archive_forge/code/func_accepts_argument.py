from __future__ import annotations
import inspect
import random
import threading
from collections import OrderedDict, UserDict
from collections.abc import Iterable, Mapping
from itertools import count, repeat
from time import sleep, time
from vine.utils import wraps
from .encoding import safe_repr as _safe_repr
def accepts_argument(func, argument_name):
    argument_spec = inspect.getfullargspec(func)
    return argument_name in argument_spec.args or argument_name in argument_spec.kwonlyargs