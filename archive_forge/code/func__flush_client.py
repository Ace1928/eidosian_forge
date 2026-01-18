import sys
from sentry_sdk.hub import Hub
from sentry_sdk.utils import event_from_exception
from sentry_sdk._compat import reraise
from sentry_sdk._functools import wraps
from sentry_sdk._types import TYPE_CHECKING
def _flush_client():
    return Hub.current.flush()