from importlib import import_module
from sentry_sdk.hub import Hub, _should_send_default_pii
from sentry_sdk.integrations import DidNotEnable, Integration
from sentry_sdk.integrations.logging import ignore_logger
from sentry_sdk.integrations._wsgi_common import request_body_within_bounds
from sentry_sdk.utils import (
from sentry_sdk._types import TYPE_CHECKING
def _sentry_patched_handle_query_result(result, *args, **kwargs):
    hub = Hub.current
    integration = hub.get_integration(AriadneIntegration)
    if integration is None:
        return old_handle_query_result(result, *args, **kwargs)
    query_result = old_handle_query_result(result, *args, **kwargs)
    with hub.configure_scope() as scope:
        event_processor = _make_response_event_processor(query_result[1])
        scope.add_event_processor(event_processor)
    if hub.client:
        with capture_internal_exceptions():
            for error in result.errors or []:
                event, hint = event_from_exception(error, client_options=hub.client.options, mechanism={'type': integration.identifier, 'handled': False})
                hub.capture_event(event, hint=hint)
    return query_result