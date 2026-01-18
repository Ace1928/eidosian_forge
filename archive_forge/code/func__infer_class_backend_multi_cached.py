import math
import importlib
import functools
import itertools
import threading
import contextlib
from inspect import signature
from collections import OrderedDict, defaultdict
@functools.lru_cache(None)
def _infer_class_backend_multi_cached(classes):
    return max(map(_infer_class_backend_cached, classes), key=lambda n: multi_class_priorities.get(n, 0))