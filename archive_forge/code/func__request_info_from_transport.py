from sentry_sdk.utils import event_from_exception, parse_version
from sentry_sdk.hub import Hub, _should_send_default_pii
from sentry_sdk.integrations import DidNotEnable, Integration
from sentry_sdk._types import TYPE_CHECKING
def _request_info_from_transport(transport):
    if transport is None:
        return {}
    request_info = {'method': _transport_method(transport)}
    try:
        request_info['url'] = transport.url
    except AttributeError:
        pass
    return request_info