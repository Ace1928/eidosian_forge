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
class HubMeta(type):

    @property
    def current(cls):
        """Returns the current instance of the hub."""
        rv = _local.get(None)
        if rv is None:
            rv = Hub(GLOBAL_HUB)
            _local.set(rv)
        return rv

    @property
    def main(cls):
        """Returns the main instance of the hub."""
        return GLOBAL_HUB