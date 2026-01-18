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
def _get_free_pos(cls, instance=None):
    """Skips specified instance."""
    positions = {abs(inst.pos) for inst in cls._instances if inst is not instance and hasattr(inst, 'pos')}
    return min(set(range(len(positions) + 1)).difference(positions))