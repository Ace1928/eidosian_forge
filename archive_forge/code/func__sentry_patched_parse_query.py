from importlib import import_module
from sentry_sdk.hub import Hub, _should_send_default_pii
from sentry_sdk.integrations import DidNotEnable, Integration
from sentry_sdk.integrations.logging import ignore_logger
from sentry_sdk.integrations._wsgi_common import request_body_within_bounds
from sentry_sdk.utils import (
from sentry_sdk._types import TYPE_CHECKING
def _sentry_patched_parse_query(context_value, query_parser, data):
    hub = Hub.current
    integration = hub.get_integration(AriadneIntegration)
    if integration is None:
        return old_parse_query(context_value, query_parser, data)
    with hub.configure_scope() as scope:
        event_processor = _make_request_event_processor(data)
        scope.add_event_processor(event_processor)
    result = old_parse_query(context_value, query_parser, data)
    return result