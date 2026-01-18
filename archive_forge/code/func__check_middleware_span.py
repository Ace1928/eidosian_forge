from django import VERSION as DJANGO_VERSION
from sentry_sdk import Hub
from sentry_sdk._functools import wraps
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.consts import OP
from sentry_sdk.utils import (
def _check_middleware_span(old_method):
    hub = Hub.current
    integration = hub.get_integration(DjangoIntegration)
    if integration is None or not integration.middleware_spans:
        return None
    function_name = transaction_from_function(old_method)
    description = middleware_name
    function_basename = getattr(old_method, '__name__', None)
    if function_basename:
        description = '{}.{}'.format(description, function_basename)
    middleware_span = hub.start_span(op=OP.MIDDLEWARE_DJANGO, description=description)
    middleware_span.set_tag('django.function_name', function_name)
    middleware_span.set_tag('django.middleware_name', middleware_name)
    return middleware_span