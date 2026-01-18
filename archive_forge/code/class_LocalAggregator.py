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
class LocalAggregator(object):
    __slots__ = ('_measurements',)

    def __init__(self):
        self._measurements = {}

    def add(self, ty, key, value, unit, tags):
        export_key = '%s:%s@%s' % (ty, key, unit)
        bucket_key = (export_key, tags)
        old = self._measurements.get(bucket_key)
        if old is not None:
            v_min, v_max, v_count, v_sum = old
            v_min = min(v_min, value)
            v_max = max(v_max, value)
            v_count += 1
            v_sum += value
        else:
            v_min = v_max = v_sum = value
            v_count = 1
        self._measurements[bucket_key] = (v_min, v_max, v_count, v_sum)

    def to_json(self):
        rv = {}
        for (export_key, tags), (v_min, v_max, v_count, v_sum) in self._measurements.items():
            rv.setdefault(export_key, []).append({'tags': _tags_to_dict(tags), 'min': v_min, 'max': v_max, 'count': v_count, 'sum': v_sum})
        return rv