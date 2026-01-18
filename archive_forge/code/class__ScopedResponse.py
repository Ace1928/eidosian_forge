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
class _ScopedResponse(object):
    __slots__ = ('_response', '_hub')

    def __init__(self, hub, response):
        self._hub = hub
        self._response = response

    def __iter__(self):
        iterator = iter(self._response)
        while True:
            with self._hub:
                try:
                    chunk = next(iterator)
                except StopIteration:
                    break
                except BaseException:
                    reraise(*_capture_exception(self._hub))
            yield chunk

    def close(self):
        with self._hub:
            try:
                self._response.close()
            except AttributeError:
                pass
            except BaseException:
                reraise(*_capture_exception(self._hub))