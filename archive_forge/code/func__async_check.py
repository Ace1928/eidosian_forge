import asyncio
from django.core.handlers.wsgi import WSGIRequest
from sentry_sdk import Hub, _functools
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.consts import OP
from sentry_sdk.hub import _should_send_default_pii
from sentry_sdk.integrations.asgi import SentryAsgiMiddleware
from sentry_sdk.utils import capture_internal_exceptions
def _async_check(self):
    """
            If get_response is a coroutine function, turns us into async mode so
            a thread is not consumed during a whole request.
            Taken from django.utils.deprecation::MiddlewareMixin._async_check
            """
    if asyncio.iscoroutinefunction(self.get_response):
        self._is_coroutine = asyncio.coroutines._is_coroutine