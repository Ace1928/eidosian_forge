from __future__ import absolute_import
import asyncio
import functools
from copy import deepcopy
from sentry_sdk._compat import iteritems
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.consts import OP
from sentry_sdk.hub import Hub, _should_send_default_pii
from sentry_sdk.integrations import DidNotEnable, Integration
from sentry_sdk.integrations._wsgi_common import (
from sentry_sdk.integrations.asgi import SentryAsgiMiddleware
from sentry_sdk.tracing import (
from sentry_sdk.utils import (
class StarletteIntegration(Integration):
    identifier = 'starlette'
    transaction_style = ''

    def __init__(self, transaction_style='url'):
        if transaction_style not in TRANSACTION_STYLE_VALUES:
            raise ValueError('Invalid value for transaction_style: %s (must be in %s)' % (transaction_style, TRANSACTION_STYLE_VALUES))
        self.transaction_style = transaction_style

    @staticmethod
    def setup_once():
        version = parse_version(STARLETTE_VERSION)
        if version is None:
            raise DidNotEnable('Unparsable Starlette version: {}'.format(STARLETTE_VERSION))
        patch_middlewares()
        patch_asgi_app()
        patch_request_response()
        if version >= (0, 24):
            patch_templates()