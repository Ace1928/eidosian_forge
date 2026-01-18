import asyncio
from django.core.handlers.wsgi import WSGIRequest
from sentry_sdk import Hub, _functools
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.consts import OP
from sentry_sdk.hub import _should_send_default_pii
from sentry_sdk.integrations.asgi import SentryAsgiMiddleware
from sentry_sdk.utils import capture_internal_exceptions

            Function that checks if we are in async mode,
            and if we are forwards the handling of requests to __acall__
            