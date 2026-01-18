import os
import subprocess
import sys
import platform
from sentry_sdk.consts import OP, SPANDATA
from sentry_sdk.hub import Hub
from sentry_sdk.integrations import Integration
from sentry_sdk.scope import add_global_event_processor
from sentry_sdk.tracing_utils import EnvironHeaders, should_propagate_trace
from sentry_sdk.utils import (
from sentry_sdk._types import TYPE_CHECKING
def _install_httplib():
    real_putrequest = HTTPConnection.putrequest
    real_getresponse = HTTPConnection.getresponse

    def putrequest(self, method, url, *args, **kwargs):
        hub = Hub.current
        host = self.host
        port = self.port
        default_port = self.default_port
        if hub.get_integration(StdlibIntegration) is None or is_sentry_url(hub, host):
            return real_putrequest(self, method, url, *args, **kwargs)
        real_url = url
        if real_url is None or not real_url.startswith(('http://', 'https://')):
            real_url = '%s://%s%s%s' % (default_port == 443 and 'https' or 'http', host, port != default_port and ':%s' % port or '', url)
        parsed_url = None
        with capture_internal_exceptions():
            parsed_url = parse_url(real_url, sanitize=False)
        span = hub.start_span(op=OP.HTTP_CLIENT, description='%s %s' % (method, parsed_url.url if parsed_url else SENSITIVE_DATA_SUBSTITUTE))
        span.set_data(SPANDATA.HTTP_METHOD, method)
        if parsed_url is not None:
            span.set_data('url', parsed_url.url)
            span.set_data(SPANDATA.HTTP_QUERY, parsed_url.query)
            span.set_data(SPANDATA.HTTP_FRAGMENT, parsed_url.fragment)
        rv = real_putrequest(self, method, url, *args, **kwargs)
        if should_propagate_trace(hub, real_url):
            for key, value in hub.iter_trace_propagation_headers(span):
                logger.debug('[Tracing] Adding `{key}` header {value} to outgoing request to {real_url}.'.format(key=key, value=value, real_url=real_url))
                self.putheader(key, value)
        self._sentrysdk_span = span
        return rv

    def getresponse(self, *args, **kwargs):
        span = getattr(self, '_sentrysdk_span', None)
        if span is None:
            return real_getresponse(self, *args, **kwargs)
        rv = real_getresponse(self, *args, **kwargs)
        span.set_http_status(int(rv.status))
        span.set_data('reason', rv.reason)
        span.finish()
        return rv
    HTTPConnection.putrequest = putrequest
    HTTPConnection.getresponse = getresponse