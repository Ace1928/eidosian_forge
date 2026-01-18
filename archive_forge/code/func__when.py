import copy
import errno
import heapq
import os
import shelve
import sys
import time
import traceback
from calendar import timegm
from collections import namedtuple
from functools import total_ordering
from threading import Event, Thread
from billiard import ensure_multiprocessing
from billiard.common import reset_signals
from billiard.context import Process
from kombu.utils.functional import maybe_evaluate, reprcall
from kombu.utils.objects import cached_property
from . import __version__, platforms, signals
from .exceptions import reraise
from .schedules import crontab, maybe_schedule
from .utils.functional import is_numeric_value
from .utils.imports import load_extension_class_names, symbol_by_name
from .utils.log import get_logger, iter_open_logger_fds
from .utils.time import humanize_seconds, maybe_make_aware
def _when(self, entry, next_time_to_run, mktime=timegm):
    """Return a utc timestamp, make sure heapq in correct order."""
    adjust = self.adjust
    as_now = maybe_make_aware(entry.default_now())
    return mktime(as_now.utctimetuple()) + as_now.microsecond / 1000000.0 + (adjust(next_time_to_run) or 0)