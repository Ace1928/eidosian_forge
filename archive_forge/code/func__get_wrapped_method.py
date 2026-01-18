from django import VERSION as DJANGO_VERSION
from sentry_sdk import Hub
from sentry_sdk._functools import wraps
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.consts import OP
from sentry_sdk.utils import (
def _get_wrapped_method(old_method):
    with capture_internal_exceptions():

        def sentry_wrapped_method(*args, **kwargs):
            middleware_span = _check_middleware_span(old_method)
            if middleware_span is None:
                return old_method(*args, **kwargs)
            with middleware_span:
                return old_method(*args, **kwargs)
        try:
            sentry_wrapped_method = wraps(old_method)(sentry_wrapped_method)
            sentry_wrapped_method.__self__ = old_method.__self__
        except Exception:
            pass
        return sentry_wrapped_method
    return old_method