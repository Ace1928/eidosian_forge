import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple
from ray.autoscaler._private.constants import (
from ray.autoscaler.node_launch_exception import NodeLaunchException
def _remove_old_entries(self):
    """Remove any expired entries from the cache."""
    cur_time = self.timer()
    with self.lock:
        for key, (expiration_time, _) in list(self.store.items()):
            if expiration_time < cur_time:
                del self.store[key]