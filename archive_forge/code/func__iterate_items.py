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
def _iterate_items(self):
    with self.mutex:
        for k in self:
            try:
                yield (k, self.data[k])
            except KeyError:
                pass