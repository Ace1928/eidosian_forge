import sys
from copy import deepcopy
from datetime import timedelta
from os import environ
from sentry_sdk.api import continue_trace
from sentry_sdk.consts import OP
from sentry_sdk.hub import Hub, _should_send_default_pii
from sentry_sdk.tracing import TRANSACTION_SOURCE_COMPONENT
from sentry_sdk.utils import (
from sentry_sdk.integrations import Integration
from sentry_sdk.integrations._wsgi_common import _filter_headers
from sentry_sdk._compat import datetime_utcnow, reraise
from sentry_sdk._types import TYPE_CHECKING
class AwsLambdaIntegration(Integration):
    identifier = 'aws_lambda'

    def __init__(self, timeout_warning=False):
        self.timeout_warning = timeout_warning

    @staticmethod
    def setup_once():
        lambda_bootstrap = get_lambda_bootstrap()
        if not lambda_bootstrap:
            logger.warning('Not running in AWS Lambda environment, AwsLambdaIntegration disabled (could not find bootstrap module)')
            return
        if not hasattr(lambda_bootstrap, 'handle_event_request'):
            logger.warning('Not running in AWS Lambda environment, AwsLambdaIntegration disabled (could not find handle_event_request)')
            return
        pre_37 = hasattr(lambda_bootstrap, 'handle_http_request')
        if pre_37:
            old_handle_event_request = lambda_bootstrap.handle_event_request

            def sentry_handle_event_request(request_handler, *args, **kwargs):
                request_handler = _wrap_handler(request_handler)
                return old_handle_event_request(request_handler, *args, **kwargs)
            lambda_bootstrap.handle_event_request = sentry_handle_event_request
            old_handle_http_request = lambda_bootstrap.handle_http_request

            def sentry_handle_http_request(request_handler, *args, **kwargs):
                request_handler = _wrap_handler(request_handler)
                return old_handle_http_request(request_handler, *args, **kwargs)
            lambda_bootstrap.handle_http_request = sentry_handle_http_request
            old_to_json = lambda_bootstrap.to_json

            def sentry_to_json(*args, **kwargs):
                _drain_queue()
                return old_to_json(*args, **kwargs)
            lambda_bootstrap.to_json = sentry_to_json
        else:
            lambda_bootstrap.LambdaRuntimeClient.post_init_error = _wrap_init_error(lambda_bootstrap.LambdaRuntimeClient.post_init_error)
            old_handle_event_request = lambda_bootstrap.handle_event_request

            def sentry_handle_event_request(lambda_runtime_client, request_handler, *args, **kwargs):
                request_handler = _wrap_handler(request_handler)
                return old_handle_event_request(lambda_runtime_client, request_handler, *args, **kwargs)
            lambda_bootstrap.handle_event_request = sentry_handle_event_request

            def _wrap_post_function(f):

                def inner(*args, **kwargs):
                    _drain_queue()
                    return f(*args, **kwargs)
                return inner
            lambda_bootstrap.LambdaRuntimeClient.post_invocation_result = _wrap_post_function(lambda_bootstrap.LambdaRuntimeClient.post_invocation_result)
            lambda_bootstrap.LambdaRuntimeClient.post_invocation_error = _wrap_post_function(lambda_bootstrap.LambdaRuntimeClient.post_invocation_error)