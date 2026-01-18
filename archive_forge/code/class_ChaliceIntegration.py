import sys
from sentry_sdk._compat import reraise
from sentry_sdk.hub import Hub
from sentry_sdk.integrations import Integration, DidNotEnable
from sentry_sdk.integrations.aws_lambda import _make_request_event_processor
from sentry_sdk.tracing import TRANSACTION_SOURCE_COMPONENT
from sentry_sdk.utils import (
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk._functools import wraps
import chalice  # type: ignore
from chalice import Chalice, ChaliceViewError
from chalice.app import EventSourceHandler as ChaliceEventSourceHandler  # type: ignore
class ChaliceIntegration(Integration):
    identifier = 'chalice'

    @staticmethod
    def setup_once():
        version = parse_version(CHALICE_VERSION)
        if version is None:
            raise DidNotEnable('Unparsable Chalice version: {}'.format(CHALICE_VERSION))
        if version < (1, 20):
            old_get_view_function_response = Chalice._get_view_function_response
        else:
            from chalice.app import RestAPIEventHandler
            old_get_view_function_response = RestAPIEventHandler._get_view_function_response

        def sentry_event_response(app, view_function, function_args):
            wrapped_view_function = _get_view_function_response(app, view_function, function_args)
            return old_get_view_function_response(app, wrapped_view_function, function_args)
        if version < (1, 20):
            Chalice._get_view_function_response = sentry_event_response
        else:
            RestAPIEventHandler._get_view_function_response = sentry_event_response
        chalice.app.EventSourceHandler = EventSourceHandler