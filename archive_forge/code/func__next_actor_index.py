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
def _next_actor_index(self):
    if self._current_index == len(self._actor_pool) - 1:
        self._current_index = 0
    else:
        self._current_index += 1
    return self._current_index