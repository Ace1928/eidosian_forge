from __future__ import absolute_import
from sentry_sdk.hub import Hub
from sentry_sdk.integrations import Integration, DidNotEnable
from sentry_sdk.integrations._wsgi_common import RequestExtractor
from sentry_sdk.integrations.wsgi import SentryWsgiMiddleware
from sentry_sdk.tracing import SOURCE_FOR_STYLE
from sentry_sdk.utils import (
from sentry_sdk._types import TYPE_CHECKING
def _patch_handle_exception():
    original_handle_exception = falcon_app_class._handle_exception

    def sentry_patched_handle_exception(self, *args):
        ex = response = None
        with capture_internal_exceptions():
            ex = next((argument for argument in args if isinstance(argument, Exception)))
            response = next((argument for argument in args if isinstance(argument, falcon.Response)))
        was_handled = original_handle_exception(self, *args)
        if ex is None or response is None:
            return was_handled
        hub = Hub.current
        integration = hub.get_integration(FalconIntegration)
        if integration is not None and _exception_leads_to_http_5xx(ex, response):
            client = hub.client
            event, hint = event_from_exception(ex, client_options=client.options, mechanism={'type': 'falcon', 'handled': False})
            hub.capture_event(event, hint=hint)
        return was_handled
    falcon_app_class._handle_exception = sentry_patched_handle_exception