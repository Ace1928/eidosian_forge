import dataclasses
import importlib
import json
import threading
import warnings
from collections import defaultdict, deque, namedtuple, OrderedDict
from typing import (
def _deque_unflatten(values: Iterable[Any], context: Context) -> Deque[Any]:
    return deque(values, maxlen=context)