import copy
import sys
from contextlib import contextmanager
from sentry_sdk._compat import with_metaclass
from sentry_sdk.consts import INSTRUMENTER
from sentry_sdk.scope import Scope
from sentry_sdk.client import Client
from sentry_sdk.tracing import (
from sentry_sdk.utils import (
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk._types import TYPE_CHECKING
class _InitGuard(object):

    def __init__(self, client):
        self._client = client

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, tb):
        c = self._client
        if c is not None:
            c.close()