import sys
from sentry_sdk._compat import PY2, reraise
from sentry_sdk._functools import partial
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk._werkzeug import get_host, _get_headers
from sentry_sdk.api import continue_trace
from sentry_sdk.consts import OP
from sentry_sdk.hub import Hub, _should_send_default_pii
from sentry_sdk.utils import (
from sentry_sdk.tracing import Transaction, TRANSACTION_SOURCE_ROUTE
from sentry_sdk.sessions import auto_session_tracking
from sentry_sdk.integrations._wsgi_common import _filter_headers
class SentryWsgiMiddleware(object):
    __slots__ = ('app', 'use_x_forwarded_for')

    def __init__(self, app, use_x_forwarded_for=False):
        self.app = app
        self.use_x_forwarded_for = use_x_forwarded_for

    def __call__(self, environ, start_response):
        if _wsgi_middleware_applied.get(False):
            return self.app(environ, start_response)
        _wsgi_middleware_applied.set(True)
        try:
            hub = Hub(Hub.current)
            with auto_session_tracking(hub, session_mode='request'):
                with hub:
                    with capture_internal_exceptions():
                        with hub.configure_scope() as scope:
                            scope.clear_breadcrumbs()
                            scope._name = 'wsgi'
                            scope.add_event_processor(_make_wsgi_event_processor(environ, self.use_x_forwarded_for))
                    transaction = continue_trace(environ, op=OP.HTTP_SERVER, name='generic WSGI request', source=TRANSACTION_SOURCE_ROUTE)
                    with hub.start_transaction(transaction, custom_sampling_context={'wsgi_environ': environ}):
                        try:
                            rv = self.app(environ, partial(_sentry_start_response, start_response, transaction))
                        except BaseException:
                            reraise(*_capture_exception(hub))
        finally:
            _wsgi_middleware_applied.set(False)
        return _ScopedResponse(hub, rv)