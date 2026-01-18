import sys
from sentry_sdk._compat import PY2, reraise
from sentry_sdk._functools import partial
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk._werkzeug import get_host, _get_headers
from sentry_sdk.api import continue_trace
from sentry_sdk.consts import OP
from sentry_sdk.hub import Hub, _should_send_default_pii
from sentry_sdk.utils import (
from sentry_sdk.tracing import Transaction, TRANSACTION_SOURCE_ROUTE
from sentry_sdk.sessions import auto_session_tracking
from sentry_sdk.integrations._wsgi_common import _filter_headers
def _sentry_start_response(old_start_response, transaction, status, response_headers, exc_info=None):
    with capture_internal_exceptions():
        status_int = int(status.split(' ', 1)[0])
        transaction.set_http_status(status_int)
    if exc_info is None:
        return old_start_response(status, response_headers)
    else:
        return old_start_response(status, response_headers, exc_info)