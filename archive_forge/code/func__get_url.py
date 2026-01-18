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
def _get_url(aws_event, aws_context):
    path = aws_event.get('path', None)
    headers = aws_event.get('headers')
    if headers is None:
        headers = {}
    host = headers.get('Host', None)
    proto = headers.get('X-Forwarded-Proto', None)
    if proto and host and path:
        return '{}://{}{}'.format(proto, host, path)
    return 'awslambda:///{}'.format(aws_context.function_name)