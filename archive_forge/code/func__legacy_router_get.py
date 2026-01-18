import sys
import weakref
from inspect import isawaitable
from sentry_sdk import continue_trace
from sentry_sdk._compat import urlparse, reraise
from sentry_sdk.consts import OP
from sentry_sdk.hub import Hub
from sentry_sdk.tracing import TRANSACTION_SOURCE_COMPONENT, TRANSACTION_SOURCE_URL
from sentry_sdk.utils import (
from sentry_sdk.integrations import Integration, DidNotEnable
from sentry_sdk.integrations._wsgi_common import RequestExtractor, _filter_headers
from sentry_sdk.integrations.logging import ignore_logger
from sentry_sdk._types import TYPE_CHECKING
def _legacy_router_get(self, *args):
    rv = old_router_get(self, *args)
    hub = Hub.current
    if hub.get_integration(SanicIntegration) is not None:
        with capture_internal_exceptions():
            with hub.configure_scope() as scope:
                if SanicIntegration.version and SanicIntegration.version >= (21, 3):
                    sanic_app_name = self.ctx.app.name
                    sanic_route = rv[0].name
                    if sanic_route.startswith('%s.' % sanic_app_name):
                        sanic_route = sanic_route[len(sanic_app_name) + 1:]
                    scope.set_transaction_name(sanic_route, source=TRANSACTION_SOURCE_COMPONENT)
                else:
                    scope.set_transaction_name(rv[0].__name__, source=TRANSACTION_SOURCE_COMPONENT)
    return rv