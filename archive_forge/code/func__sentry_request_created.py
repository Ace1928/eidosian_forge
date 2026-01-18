from __future__ import absolute_import
from sentry_sdk import Hub
from sentry_sdk.consts import OP, SPANDATA
from sentry_sdk.integrations import Integration, DidNotEnable
from sentry_sdk.tracing import Span
from sentry_sdk._functools import partial
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.utils import capture_internal_exceptions, parse_url, parse_version
def _sentry_request_created(service_id, request, operation_name, **kwargs):
    hub = Hub.current
    if hub.get_integration(Boto3Integration) is None:
        return
    description = 'aws.%s.%s' % (service_id, operation_name)
    span = hub.start_span(hub=hub, op=OP.HTTP_CLIENT, description=description)
    with capture_internal_exceptions():
        parsed_url = parse_url(request.url, sanitize=False)
        span.set_data('aws.request.url', parsed_url.url)
        span.set_data(SPANDATA.HTTP_QUERY, parsed_url.query)
        span.set_data(SPANDATA.HTTP_FRAGMENT, parsed_url.fragment)
    span.set_tag('aws.service_id', service_id)
    span.set_tag('aws.operation_name', operation_name)
    span.set_data(SPANDATA.HTTP_METHOD, request.method)
    span.__enter__()
    request.context['_sentrysdk_span'] = span