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
class DistributionMetric(Metric):
    __slots__ = ('value',)

    def __init__(self, first):
        self.value = [float(first)]

    @property
    def weight(self):
        return len(self.value)

    def add(self, value):
        self.value.append(float(value))

    def serialize_value(self):
        return self.value