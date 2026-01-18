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
class CallableDefaultdict(defaultdict):
    """:class:`~collections.defaultdict` with configurable __call__.

    We use this for backwards compatibility in State.tasks_by_type
    etc, which used to be a method but is now an index instead.

    So you can do::

        >>> add_tasks = state.tasks_by_type['proj.tasks.add']

    while still supporting the method call::

        >>> add_tasks = list(state.tasks_by_type(
        ...     'proj.tasks.add', reverse=True))
    """

    def __init__(self, fun, *args, **kwargs):
        self.fun = fun
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.fun(*args, **kwargs)