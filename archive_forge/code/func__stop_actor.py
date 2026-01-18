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
def _stop_actor(self, actor):
    self._wait_for_stopping_actors(timeout=0.0)
    self._actor_deletion_ids.append(actor.__ray_terminate__.remote())