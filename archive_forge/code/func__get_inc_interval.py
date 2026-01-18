import functools
import logging
import random
import threading
import time
from oslo_utils import excutils
from oslo_utils import importutils
from oslo_utils import reflection
from oslo_db import exception
from oslo_db import options
def _get_inc_interval(self, n, jitter):
    n = n * 2
    if jitter:
        sleep_time = random.uniform(0, n)
    else:
        sleep_time = n
    return (min(sleep_time, self.max_retry_interval), n)