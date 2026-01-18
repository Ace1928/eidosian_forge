import io
import os
import random
import re
import sys
import threading
import time
import zlib
from contextlib import contextmanager
from datetime import datetime
from functools import wraps, partial
import sentry_sdk
from sentry_sdk._compat import text_type, utc_from_timestamp, iteritems
from sentry_sdk.utils import (
from sentry_sdk.envelope import Envelope, Item
from sentry_sdk.tracing import (
from sentry_sdk._types import TYPE_CHECKING
class _Timing(object):

    def __init__(self, key, tags, timestamp, value, unit, stacklevel):
        self.key = key
        self.tags = tags
        self.timestamp = timestamp
        self.value = value
        self.unit = unit
        self.entered = None
        self._span = None
        self.stacklevel = stacklevel

    def _validate_invocation(self, context):
        if self.value is not None:
            raise TypeError('cannot use timing as %s when a value is provided' % context)

    def __enter__(self):
        self.entered = TIMING_FUNCTIONS[self.unit]()
        self._validate_invocation('context-manager')
        self._span = sentry_sdk.start_span(op='metric.timing', description=self.key)
        if self.tags:
            for key, value in self.tags.items():
                if isinstance(value, (tuple, list)):
                    value = ','.join(sorted(map(str, value)))
                self._span.set_tag(key, value)
        self._span.__enter__()
        aggregator = _get_aggregator()
        if aggregator is not None:
            aggregator.record_code_location('d', self.key, self.unit, self.stacklevel)
        return self

    def __exit__(self, exc_type, exc_value, tb):
        assert self._span, 'did not enter'
        aggregator, local_aggregator, tags = _get_aggregator_and_update_tags(self.key, self.tags)
        if aggregator is not None:
            elapsed = TIMING_FUNCTIONS[self.unit]() - self.entered
            aggregator.add('d', self.key, elapsed, self.unit, tags, self.timestamp, local_aggregator, None)
        self._span.__exit__(exc_type, exc_value, tb)
        self._span = None

    def __call__(self, f):
        self._validate_invocation('decorator')

        @wraps(f)
        def timed_func(*args, **kwargs):
            with timing(key=self.key, tags=self.tags, timestamp=self.timestamp, unit=self.unit, stacklevel=self.stacklevel + 1):
                return f(*args, **kwargs)
        return timed_func