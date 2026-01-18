import sys
import weakref
from inspect import isawaitable
from sentry_sdk import continue_trace
from sentry_sdk._compat import urlparse, reraise
from sentry_sdk.consts import OP
from sentry_sdk.hub import Hub
from sentry_sdk.tracing import TRANSACTION_SOURCE_COMPONENT, TRANSACTION_SOURCE_URL
from sentry_sdk.utils import (
from sentry_sdk.integrations import Integration, DidNotEnable
from sentry_sdk.integrations._wsgi_common import RequestExtractor, _filter_headers
from sentry_sdk.integrations.logging import ignore_logger
from sentry_sdk._types import TYPE_CHECKING
def _sentry_error_handler_lookup(self, exception, *args, **kwargs):
    _capture_exception(exception)
    old_error_handler = old_error_handler_lookup(self, exception, *args, **kwargs)
    if old_error_handler is None:
        return None
    if Hub.current.get_integration(SanicIntegration) is None:
        return old_error_handler

    async def sentry_wrapped_error_handler(request, exception):
        try:
            response = old_error_handler(request, exception)
            if isawaitable(response):
                response = await response
            return response
        except Exception:
            exc_info = sys.exc_info()
            _capture_exception(exc_info)
            reraise(*exc_info)
        finally:
            if SanicIntegration.version and SanicIntegration.version == (21, 9):
                await _hub_exit(request)
    return sentry_wrapped_error_handler