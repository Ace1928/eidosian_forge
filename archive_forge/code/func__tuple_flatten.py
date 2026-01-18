import dataclasses
import importlib
import json
import threading
import warnings
from collections import defaultdict, deque, namedtuple, OrderedDict
from typing import (
def _tuple_flatten(d: Tuple[Any, ...]) -> Tuple[List[Any], Context]:
    return (list(d), None)