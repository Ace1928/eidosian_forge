from __future__ import absolute_import
import sys
from functools import wraps
from threading import Thread, current_thread
from sentry_sdk import Hub
from sentry_sdk._compat import reraise
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.integrations import Integration
from sentry_sdk.utils import event_from_exception, capture_internal_exceptions
class ThreadingIntegration(Integration):
    identifier = 'threading'

    def __init__(self, propagate_hub=False):
        self.propagate_hub = propagate_hub

    @staticmethod
    def setup_once():
        old_start = Thread.start

        @wraps(old_start)
        def sentry_start(self, *a, **kw):
            hub = Hub.current
            integration = hub.get_integration(ThreadingIntegration)
            if integration is not None:
                if not integration.propagate_hub:
                    hub_ = None
                else:
                    hub_ = Hub(hub)
                with capture_internal_exceptions():
                    new_run = _wrap_run(hub_, getattr(self.run, '__func__', self.run))
                    self.run = new_run
            return old_start(self, *a, **kw)
        Thread.start = sentry_start