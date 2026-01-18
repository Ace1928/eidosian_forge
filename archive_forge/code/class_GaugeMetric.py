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
class GaugeMetric(Metric):
    __slots__ = ('last', 'min', 'max', 'sum', 'count')

    def __init__(self, first):
        first = float(first)
        self.last = first
        self.min = first
        self.max = first
        self.sum = first
        self.count = 1

    @property
    def weight(self):
        return 5

    def add(self, value):
        value = float(value)
        self.last = value
        self.min = min(self.min, value)
        self.max = max(self.max, value)
        self.sum += value
        self.count += 1

    def serialize_value(self):
        return (self.last, self.min, self.max, self.sum, self.count)