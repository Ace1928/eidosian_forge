import sys
import weakref
from inspect import isawaitable
from sentry_sdk import continue_trace
from sentry_sdk._compat import urlparse, reraise
from sentry_sdk.consts import OP
from sentry_sdk.hub import Hub
from sentry_sdk.tracing import TRANSACTION_SOURCE_COMPONENT, TRANSACTION_SOURCE_URL
from sentry_sdk.utils import (
from sentry_sdk.integrations import Integration, DidNotEnable
from sentry_sdk.integrations._wsgi_common import RequestExtractor, _filter_headers
from sentry_sdk.integrations.logging import ignore_logger
from sentry_sdk._types import TYPE_CHECKING
class SanicIntegration(Integration):
    identifier = 'sanic'
    version = None

    def __init__(self, unsampled_statuses=frozenset({404})):
        """
        The unsampled_statuses parameter can be used to specify for which HTTP statuses the
        transactions should not be sent to Sentry. By default, transactions are sent for all
        HTTP statuses, except 404. Set unsampled_statuses to None to send transactions for all
        HTTP statuses, including 404.
        """
        self._unsampled_statuses = unsampled_statuses or set()

    @staticmethod
    def setup_once():
        SanicIntegration.version = parse_version(SANIC_VERSION)
        if SanicIntegration.version is None:
            raise DidNotEnable('Unparsable Sanic version: {}'.format(SANIC_VERSION))
        if SanicIntegration.version < (0, 8):
            raise DidNotEnable('Sanic 0.8 or newer required.')
        if not HAS_REAL_CONTEXTVARS:
            raise DidNotEnable('The sanic integration for Sentry requires Python 3.7+  or the aiocontextvars package.' + CONTEXTVARS_ERROR_MESSAGE)
        if SANIC_VERSION.startswith('0.8.'):
            ignore_logger('root')
        if SanicIntegration.version < (21, 9):
            _setup_legacy_sanic()
            return
        _setup_sanic()