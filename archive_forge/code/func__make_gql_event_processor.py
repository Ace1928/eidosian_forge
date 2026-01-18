from sentry_sdk.utils import event_from_exception, parse_version
from sentry_sdk.hub import Hub, _should_send_default_pii
from sentry_sdk.integrations import DidNotEnable, Integration
from sentry_sdk._types import TYPE_CHECKING
def _make_gql_event_processor(client, document):

    def processor(event, hint):
        try:
            errors = hint['exc_info'][1].errors
        except (AttributeError, KeyError):
            errors = None
        request = event.setdefault('request', {})
        request.update({'api_target': 'graphql', **_request_info_from_transport(client.transport)})
        if _should_send_default_pii():
            request['data'] = _data_from_document(document)
            contexts = event.setdefault('contexts', {})
            response = contexts.setdefault('response', {})
            response.update({'data': {'errors': errors}, 'type': response})
        return event
    return processor