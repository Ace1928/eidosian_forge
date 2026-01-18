import sys
from sentry_sdk.hub import Hub
from sentry_sdk.utils import event_from_exception
from sentry_sdk._compat import reraise
from sentry_sdk._functools import wraps
from sentry_sdk._types import TYPE_CHECKING
def _capture_and_reraise():
    exc_info = sys.exc_info()
    hub = Hub.current
    if hub.client is not None:
        event, hint = event_from_exception(exc_info, client_options=hub.client.options, mechanism={'type': 'serverless', 'handled': False})
        hub.capture_event(event, hint=hint)
    reraise(*exc_info)