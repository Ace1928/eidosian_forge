import collections
import copy
import gc
import itertools
import logging
import os
import queue
import sys
import threading
import time
from multiprocessing import TimeoutError
from typing import Any, Callable, Dict, Hashable, Iterable, List, Optional, Tuple
import ray
from ray._private.usage import usage_lib
from ray.util import log_once
def _put_in_list_registry(obj: Any, registry: List[Tuple[Any, ray.ObjectRef]]) -> ray.ObjectRef:
    try:
        ret = next((ref for o, ref in registry if o is obj))
    except StopIteration:
        ret = ray.put(obj)
        registry.append((obj, ret))
    return ret