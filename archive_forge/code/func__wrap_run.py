from __future__ import absolute_import
import sys
from functools import wraps
from threading import Thread, current_thread
from sentry_sdk import Hub
from sentry_sdk._compat import reraise
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.integrations import Integration
from sentry_sdk.utils import event_from_exception, capture_internal_exceptions
def _wrap_run(parent_hub, old_run_func):

    @wraps(old_run_func)
    def run(*a, **kw):
        hub = parent_hub or Hub.current
        with hub:
            try:
                self = current_thread()
                return old_run_func(self, *a, **kw)
            except Exception:
                reraise(*_capture_exception())
    return run