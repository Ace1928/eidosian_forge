from __future__ import absolute_import
import os
import sys
import weakref
from sentry_sdk.hub import Hub, _should_send_default_pii
from sentry_sdk.scope import Scope
from sentry_sdk.tracing import SOURCE_FOR_STYLE
from sentry_sdk.utils import (
from sentry_sdk._compat import reraise, iteritems
from sentry_sdk.integrations import Integration, DidNotEnable
from sentry_sdk.integrations._wsgi_common import RequestExtractor
from sentry_sdk.integrations.wsgi import SentryWsgiMiddleware
from sentry_sdk._types import TYPE_CHECKING
class PyramidIntegration(Integration):
    identifier = 'pyramid'
    transaction_style = ''

    def __init__(self, transaction_style='route_name'):
        if transaction_style not in TRANSACTION_STYLE_VALUES:
            raise ValueError('Invalid value for transaction_style: %s (must be in %s)' % (transaction_style, TRANSACTION_STYLE_VALUES))
        self.transaction_style = transaction_style

    @staticmethod
    def setup_once():
        from pyramid import router
        old_call_view = router._call_view

        def sentry_patched_call_view(registry, request, *args, **kwargs):
            hub = Hub.current
            integration = hub.get_integration(PyramidIntegration)
            if integration is not None:
                with hub.configure_scope() as scope:
                    _set_transaction_name_and_source(scope, integration.transaction_style, request)
                    scope.add_event_processor(_make_event_processor(weakref.ref(request), integration))
            return old_call_view(registry, request, *args, **kwargs)
        router._call_view = sentry_patched_call_view
        if hasattr(Request, 'invoke_exception_view'):
            old_invoke_exception_view = Request.invoke_exception_view

            def sentry_patched_invoke_exception_view(self, *args, **kwargs):
                rv = old_invoke_exception_view(self, *args, **kwargs)
                if self.exc_info and all(self.exc_info) and (rv.status_int == 500) and (Hub.current.get_integration(PyramidIntegration) is not None):
                    _capture_exception(self.exc_info)
                return rv
            Request.invoke_exception_view = sentry_patched_invoke_exception_view
        old_wsgi_call = router.Router.__call__

        def sentry_patched_wsgi_call(self, environ, start_response):
            hub = Hub.current
            integration = hub.get_integration(PyramidIntegration)
            if integration is None:
                return old_wsgi_call(self, environ, start_response)

            def sentry_patched_inner_wsgi_call(environ, start_response):
                try:
                    return old_wsgi_call(self, environ, start_response)
                except Exception:
                    einfo = sys.exc_info()
                    _capture_exception(einfo)
                    reraise(*einfo)
            return SentryWsgiMiddleware(sentry_patched_inner_wsgi_call)(environ, start_response)
        router.Router.__call__ = sentry_patched_wsgi_call