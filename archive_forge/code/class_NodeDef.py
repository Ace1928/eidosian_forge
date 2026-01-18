import dataclasses
import importlib
import json
import threading
import warnings
from collections import defaultdict, deque, namedtuple, OrderedDict
from typing import (
class NodeDef(NamedTuple):
    type: Type[Any]
    flatten_fn: FlattenFunc
    unflatten_fn: UnflattenFunc