import bisect
import sys
import threading
from collections import defaultdict
from collections.abc import Callable
from datetime import datetime
from decimal import Decimal
from itertools import islice
from operator import itemgetter
from time import time
from typing import Mapping, Optional  # noqa
from weakref import WeakSet, ref
from kombu.clocks import timetuple
from kombu.utils.objects import cached_property
from celery import states
from celery.utils.functional import LRUCache, memoize, pass1
from celery.utils.log import get_logger
def _tasks_by_type(self, name, limit=None, reverse=True):
    """Get all tasks by type.

        This is slower than accessing :attr:`tasks_by_type`,
        but will be ordered by time.

        Returns:
            Generator: giving ``(uuid, Task)`` pairs.
        """
    return islice(((uuid, task) for uuid, task in self.tasks_by_time(reverse=reverse) if task.name == name), 0, limit)