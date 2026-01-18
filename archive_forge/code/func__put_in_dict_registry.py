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
def _put_in_dict_registry(obj: Any, registry_hashable: Dict[Hashable, ray.ObjectRef]) -> ray.ObjectRef:
    if obj not in registry_hashable:
        ret = ray.put(obj)
        registry_hashable[obj] = ret
    else:
        ret = registry_hashable[obj]
    return ret