import random
import threading
import time
import typing
from typing import Any
from typing import Mapping
import warnings
from ..api import CacheBackend
from ..api import NO_VALUE
from ... import util
class RepairBMemcachedAPI(bmemcached.Client):
    """Repairs BMemcached's non-standard method
            signatures, which was fixed in BMemcached
            ef206ed4473fec3b639e.

            """

    def add(self, key, value, timeout=0):
        try:
            return super(RepairBMemcachedAPI, self).add(key, value, timeout)
        except ValueError:
            return False