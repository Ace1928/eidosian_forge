from __future__ import absolute_import
import asyncio
import inspect
import threading
from sentry_sdk.hub import _should_send_default_pii, Hub
from sentry_sdk.integrations import DidNotEnable, Integration
from sentry_sdk.integrations._wsgi_common import _filter_headers
from sentry_sdk.integrations.asgi import SentryAsgiMiddleware
from sentry_sdk.scope import Scope
from sentry_sdk.tracing import SOURCE_FOR_STYLE
from sentry_sdk.utils import (
from sentry_sdk._functools import wraps
from sentry_sdk._types import TYPE_CHECKING
class QuartIntegration(Integration):
    identifier = 'quart'
    transaction_style = ''

    def __init__(self, transaction_style='endpoint'):
        if transaction_style not in TRANSACTION_STYLE_VALUES:
            raise ValueError('Invalid value for transaction_style: %s (must be in %s)' % (transaction_style, TRANSACTION_STYLE_VALUES))
        self.transaction_style = transaction_style

    @staticmethod
    def setup_once():
        request_started.connect(_request_websocket_started)
        websocket_started.connect(_request_websocket_started)
        got_background_exception.connect(_capture_exception)
        got_request_exception.connect(_capture_exception)
        got_websocket_exception.connect(_capture_exception)
        patch_asgi_app()
        patch_scaffold_route()