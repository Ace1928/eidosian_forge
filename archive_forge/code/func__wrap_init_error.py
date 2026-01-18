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
def _wrap_init_error(init_error):

    def sentry_init_error(*args, **kwargs):
        hub = Hub.current
        integration = hub.get_integration(AwsLambdaIntegration)
        if integration is None:
            return init_error(*args, **kwargs)
        client = hub.client
        with capture_internal_exceptions():
            with hub.configure_scope() as scope:
                scope.clear_breadcrumbs()
            exc_info = sys.exc_info()
            if exc_info and all(exc_info):
                sentry_event, hint = event_from_exception(exc_info, client_options=client.options, mechanism={'type': 'aws_lambda', 'handled': False})
                hub.capture_event(sentry_event, hint=hint)
        return init_error(*args, **kwargs)
    return sentry_init_error