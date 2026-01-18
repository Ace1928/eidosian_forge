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
def _new_actor_entry(self):
    if not self._pool_actor:
        self._pool_actor = PoolActor.options(**self._ray_remote_args)
    return (self._pool_actor.remote(self._initializer, self._initargs), 0)