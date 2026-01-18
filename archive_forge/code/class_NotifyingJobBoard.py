import abc
import collections
import contextlib
import functools
import time
import enum
from oslo_utils import timeutils
from oslo_utils import uuidutils
import tenacity
from taskflow import exceptions as excp
from taskflow import states
from taskflow.types import notifier
from taskflow.utils import iter_utils
class NotifyingJobBoard(JobBoard):
    """A jobboard subclass that can notify others about board events.

    Implementers are expected to notify *at least* about jobs being posted
    and removed.

    NOTE(harlowja): notifications that are emitted *may* be emitted on a
    separate dedicated thread when they occur, so ensure that all callbacks
    registered are thread safe (and block for as little time as possible).
    """

    def __init__(self, name, conf):
        super(NotifyingJobBoard, self).__init__(name, conf)
        self.notifier = notifier.Notifier()