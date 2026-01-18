from the broker, processing the messages and keeping the broker connections
import errno
import logging
import os
import warnings
from collections import defaultdict
from time import sleep
from billiard.common import restart_state
from billiard.exceptions import RestartFreqExceeded
from kombu.asynchronous.semaphore import DummyLock
from kombu.exceptions import ContentDisallowed, DecodeError
from kombu.utils.compat import _detect_environment
from kombu.utils.encoding import safe_repr
from kombu.utils.limits import TokenBucket
from vine import ppartial, promise
from celery import bootsteps, signals
from celery.app.trace import build_tracer
from celery.exceptions import (CPendingDeprecationWarning, InvalidTaskError, NotRegistered, WorkerShutdown,
from celery.utils.functional import noop
from celery.utils.log import get_logger
from celery.utils.nodenames import gethostname
from celery.utils.objects import Bunch
from celery.utils.text import truncate
from celery.utils.time import humanize_seconds, rate
from celery.worker import loops
from celery.worker.state import active_requests, maybe_shutdown, requests, reserved_requests, task_reserved
def _schedule_bucket_request(self, bucket):
    while True:
        try:
            request, tokens = bucket.pop()
        except IndexError:
            break
        if bucket.can_consume(tokens):
            self._limit_move_to_pool(request)
            continue
        else:
            bucket.contents.appendleft((request, tokens))
            pri = self._limit_order = (self._limit_order + 1) % 10
            hold = bucket.expected_time(tokens)
            self.timer.call_after(hold, self._schedule_bucket_request, (bucket,), priority=pri)
            break