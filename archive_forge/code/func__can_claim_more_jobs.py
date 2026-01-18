import abc
import contextlib
import functools
import itertools
import threading
from oslo_utils import excutils
from oslo_utils import timeutils
from taskflow.conductors import base
from taskflow import exceptions as excp
from taskflow.listeners import logging as logging_listener
from taskflow import logging
from taskflow import states
from taskflow.types import timing as tt
from taskflow.utils import iter_utils
from taskflow.utils import misc
def _can_claim_more_jobs(self, job):
    if self._wait_timeout.is_stopped():
        return False
    if self._max_simultaneous_jobs <= 0:
        return True
    if len(self._dispatched) >= self._max_simultaneous_jobs:
        return False
    else:
        return True