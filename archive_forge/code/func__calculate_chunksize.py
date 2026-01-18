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
def _calculate_chunksize(self, iterable):
    chunksize, extra = divmod(len(iterable), len(self._actor_pool) * 4)
    if extra:
        chunksize += 1
    return chunksize