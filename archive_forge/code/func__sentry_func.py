from __future__ import absolute_import
import asyncio
import inspect
import threading
from sentry_sdk.hub import _should_send_default_pii, Hub
from sentry_sdk.integrations import DidNotEnable, Integration
from sentry_sdk.integrations._wsgi_common import _filter_headers
from sentry_sdk.integrations.asgi import SentryAsgiMiddleware
from sentry_sdk.scope import Scope
from sentry_sdk.tracing import SOURCE_FOR_STYLE
from sentry_sdk.utils import (
from sentry_sdk._functools import wraps
from sentry_sdk._types import TYPE_CHECKING
@wraps(old_func)
def _sentry_func(*args, **kwargs):
    hub = Hub.current
    integration = hub.get_integration(QuartIntegration)
    if integration is None:
        return old_func(*args, **kwargs)
    with hub.configure_scope() as sentry_scope:
        if sentry_scope.profile is not None:
            sentry_scope.profile.active_thread_id = threading.current_thread().ident
        return old_func(*args, **kwargs)