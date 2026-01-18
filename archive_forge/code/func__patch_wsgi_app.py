from __future__ import absolute_import
from sentry_sdk.hub import Hub
from sentry_sdk.integrations import Integration, DidNotEnable
from sentry_sdk.integrations._wsgi_common import RequestExtractor
from sentry_sdk.integrations.wsgi import SentryWsgiMiddleware
from sentry_sdk.tracing import SOURCE_FOR_STYLE
from sentry_sdk.utils import (
from sentry_sdk._types import TYPE_CHECKING
def _patch_wsgi_app():
    original_wsgi_app = falcon_app_class.__call__

    def sentry_patched_wsgi_app(self, env, start_response):
        hub = Hub.current
        integration = hub.get_integration(FalconIntegration)
        if integration is None:
            return original_wsgi_app(self, env, start_response)
        sentry_wrapped = SentryWsgiMiddleware(lambda envi, start_resp: original_wsgi_app(self, envi, start_resp))
        return sentry_wrapped(env, start_response)
    falcon_app_class.__call__ = sentry_patched_wsgi_app