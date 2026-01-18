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
def _add_object_ref(self, object_ref):
    self._indices[object_ref] = len(self._object_refs)
    self._object_refs.append(object_ref)
    self._results.append(None)