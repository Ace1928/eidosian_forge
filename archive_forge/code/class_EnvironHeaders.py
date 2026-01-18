import contextlib
import os
import re
import sys
import sentry_sdk
from sentry_sdk.consts import OP, SPANDATA
from sentry_sdk.utils import (
from sentry_sdk._compat import PY2, duration_in_milliseconds, iteritems
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.tracing import LOW_QUALITY_TRANSACTION_SOURCES
class EnvironHeaders(Mapping):

    def __init__(self, environ, prefix='HTTP_'):
        self.environ = environ
        self.prefix = prefix

    def __getitem__(self, key):
        return self.environ[self.prefix + key.replace('-', '_').upper()]

    def __len__(self):
        return sum((1 for _ in iter(self)))

    def __iter__(self):
        for k in self.environ:
            if not isinstance(k, str):
                continue
            k = k.replace('-', '_').upper()
            if not k.startswith(self.prefix):
                continue
            yield k[len(self.prefix):]