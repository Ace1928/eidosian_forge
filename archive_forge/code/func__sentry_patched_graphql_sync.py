from sentry_sdk.hub import Hub, _should_send_default_pii
from sentry_sdk.integrations import DidNotEnable, Integration
from sentry_sdk.utils import (
from sentry_sdk._types import TYPE_CHECKING
def _sentry_patched_graphql_sync(schema, source, *args, **kwargs):
    hub = Hub.current
    integration = hub.get_integration(GrapheneIntegration)
    if integration is None:
        return old_graphql_sync(schema, source, *args, **kwargs)
    with hub.configure_scope() as scope:
        scope.add_event_processor(_event_processor)
    result = old_graphql_sync(schema, source, *args, **kwargs)
    with capture_internal_exceptions():
        for error in result.errors or []:
            event, hint = event_from_exception(error, client_options=hub.client.options if hub.client else None, mechanism={'type': integration.identifier, 'handled': False})
            hub.capture_event(event, hint=hint)
    return result