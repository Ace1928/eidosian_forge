from __future__ import absolute_import
from sentry_sdk.hub import Hub
from sentry_sdk.integrations import Integration, DidNotEnable
from sentry_sdk.integrations._wsgi_common import RequestExtractor
from sentry_sdk.integrations.wsgi import SentryWsgiMiddleware
from sentry_sdk.tracing import SOURCE_FOR_STYLE
from sentry_sdk.utils import (
from sentry_sdk._types import TYPE_CHECKING
def _exception_leads_to_http_5xx(ex, response):
    is_server_error = isinstance(ex, falcon.HTTPError) and (ex.status or '').startswith('5')
    is_unhandled_error = not isinstance(ex, (falcon.HTTPError, falcon.http_status.HTTPStatus))
    return (is_server_error or is_unhandled_error) and (not FALCON3 or _has_http_5xx_status(response))