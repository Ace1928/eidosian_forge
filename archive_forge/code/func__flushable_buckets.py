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
def _flushable_buckets(self):
    with self._lock:
        force_flush = self._force_flush
        cutoff = time.time() - self.ROLLUP_IN_SECONDS - self._flush_shift
        flushable_buckets = ()
        weight_to_remove = 0
        if force_flush:
            flushable_buckets = self.buckets.items()
            self.buckets = {}
            self._buckets_total_weight = 0
            self._force_flush = False
        else:
            flushable_buckets = []
            for buckets_timestamp, buckets in iteritems(self.buckets):
                if buckets_timestamp <= cutoff:
                    flushable_buckets.append((buckets_timestamp, buckets))
            for buckets_timestamp, buckets in flushable_buckets:
                for _, metric in iteritems(buckets):
                    weight_to_remove += metric.weight
                del self.buckets[buckets_timestamp]
            self._buckets_total_weight -= weight_to_remove
    return flushable_buckets