import asyncio
import inspect
from copy import deepcopy
from sentry_sdk._functools import partial
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.api import continue_trace
from sentry_sdk.consts import OP
from sentry_sdk.hub import Hub
from sentry_sdk.integrations._asgi_common import (
from sentry_sdk.sessions import auto_session_tracking
from sentry_sdk.tracing import (
from sentry_sdk.utils import (
from sentry_sdk.tracing import Transaction
def _get_transaction_name_and_source(self, transaction_style, asgi_scope):
    name = None
    source = SOURCE_FOR_STYLE[transaction_style]
    ty = asgi_scope.get('type')
    if transaction_style == 'endpoint':
        endpoint = asgi_scope.get('endpoint')
        if endpoint:
            name = transaction_from_function(endpoint) or ''
        else:
            name = _get_url(asgi_scope, 'http' if ty == 'http' else 'ws', host=None)
            source = TRANSACTION_SOURCE_URL
    elif transaction_style == 'url':
        route = asgi_scope.get('route')
        if route:
            path = getattr(route, 'path', None)
            if path is not None:
                name = path
        else:
            name = _get_url(asgi_scope, 'http' if ty == 'http' else 'ws', host=None)
            source = TRANSACTION_SOURCE_URL
    if name is None:
        name = _DEFAULT_TRANSACTION_NAME
        source = TRANSACTION_SOURCE_ROUTE
        return (name, source)
    return (name, source)