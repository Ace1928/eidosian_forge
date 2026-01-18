import dataclasses
import importlib
import json
import threading
import warnings
from collections import defaultdict, deque, namedtuple, OrderedDict
from typing import (
def _dict_unflatten(values: Iterable[Any], context: Context) -> Dict[Any, Any]:
    return dict(zip(context, values))