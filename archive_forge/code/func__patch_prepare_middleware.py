from __future__ import absolute_import
from sentry_sdk.hub import Hub
from sentry_sdk.integrations import Integration, DidNotEnable
from sentry_sdk.integrations._wsgi_common import RequestExtractor
from sentry_sdk.integrations.wsgi import SentryWsgiMiddleware
from sentry_sdk.tracing import SOURCE_FOR_STYLE
from sentry_sdk.utils import (
from sentry_sdk._types import TYPE_CHECKING
def _patch_prepare_middleware():
    original_prepare_middleware = falcon_helpers.prepare_middleware

    def sentry_patched_prepare_middleware(middleware=None, independent_middleware=False, asgi=False):
        if asgi:
            return original_prepare_middleware(middleware, independent_middleware, asgi)
        hub = Hub.current
        integration = hub.get_integration(FalconIntegration)
        if integration is not None:
            middleware = [SentryFalconMiddleware()] + (middleware or [])
        return original_prepare_middleware(middleware, independent_middleware)
    falcon_helpers.prepare_middleware = sentry_patched_prepare_middleware