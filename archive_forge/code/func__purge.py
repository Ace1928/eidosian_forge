import math
import threading
import time
from typing import Dict, List
import ray
from ray.data.context import DataContext
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
def _purge(self):
    now = time.time()
    for k, (_, t) in list(self._resource_requests.items()):
        if t < now:
            self._resource_requests.pop(k)