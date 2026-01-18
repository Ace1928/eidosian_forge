import abc
import logging
import os
import random
import shutil
import time
import urllib
import uuid
from collections import namedtuple
from typing import IO, List, Optional, Tuple
import ray
from ray._private.ray_constants import DEFAULT_OBJECT_PREFIX
from ray._raylet import ObjectRef
class SlowFileStorage(FileSystemStorage):
    """This class is for testing slow object spilling."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._min_delay = 1
        self._max_delay = 2

    def spill_objects(self, object_refs, owner_addresses) -> List[str]:
        delay = random.random() * (self._max_delay - self._min_delay) + self._min_delay
        time.sleep(delay)
        return super().spill_objects(object_refs, owner_addresses)