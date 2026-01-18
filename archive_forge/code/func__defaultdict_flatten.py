import dataclasses
import importlib
import json
import threading
import warnings
from collections import defaultdict, deque, namedtuple, OrderedDict
from typing import (
def _defaultdict_flatten(d: DefaultDict[Any, Any]) -> Tuple[List[Any], Context]:
    values, dict_context = _dict_flatten(d)
    return (values, [d.default_factory, dict_context])