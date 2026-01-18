import math
import importlib
import functools
import itertools
import threading
import contextlib
from inspect import signature
from collections import OrderedDict, defaultdict
@functools.lru_cache(None)
def _torch_get_dtype_name(dtype):
    return str(dtype).split('.')[-1]