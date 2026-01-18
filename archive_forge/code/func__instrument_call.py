import functools
from typing import TYPE_CHECKING
from django import VERSION as DJANGO_VERSION
from django.core.cache import CacheHandler
from sentry_sdk import Hub
from sentry_sdk.consts import OP, SPANDATA
from sentry_sdk._compat import text_type
def _instrument_call(cache, method_name, original_method, args, kwargs):
    hub = Hub.current
    integration = hub.get_integration(DjangoIntegration)
    if integration is None or not integration.cache_spans:
        return original_method(*args, **kwargs)
    description = _get_span_description(method_name, args, kwargs)
    with hub.start_span(op=OP.CACHE_GET_ITEM, description=description) as span:
        value = original_method(*args, **kwargs)
        if value:
            span.set_data(SPANDATA.CACHE_HIT, True)
            size = len(text_type(value))
            span.set_data(SPANDATA.CACHE_ITEM_SIZE, size)
        else:
            span.set_data(SPANDATA.CACHE_HIT, False)
        return value