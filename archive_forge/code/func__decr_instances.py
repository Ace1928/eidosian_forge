import sys
from collections import OrderedDict, defaultdict
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from numbers import Number
from time import time
from warnings import warn
from weakref import WeakSet
from ._monitor import TMonitor
from .utils import (
@classmethod
def _decr_instances(cls, instance):
    """
        Remove from list and reposition another unfixed bar
        to fill the new gap.

        This means that by default (where all nested bars are unfixed),
        order is not maintained but screen flicker/blank space is minimised.
        (tqdm<=4.44.1 moved ALL subsequent unfixed bars up.)
        """
    with cls._lock:
        try:
            cls._instances.remove(instance)
        except KeyError:
            pass
        if not instance.gui:
            last = (instance.nrows or 20) - 1
            instances = list(filter(lambda i: hasattr(i, 'pos') and last <= i.pos, cls._instances))
            if instances:
                inst = min(instances, key=lambda i: i.pos)
                inst.clear(nolock=True)
                inst.pos = abs(instance.pos)