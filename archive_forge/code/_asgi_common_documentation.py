import urllib
from sentry_sdk.hub import _should_send_default_pii
from sentry_sdk.integrations._wsgi_common import _filter_headers
from sentry_sdk._types import TYPE_CHECKING

    Returns data related to the HTTP request from the ASGI scope.
    