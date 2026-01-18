from sentry_sdk import Hub
from sentry_sdk.consts import OP, SPANDATA
from sentry_sdk.hub import _should_send_default_pii
from sentry_sdk.integrations import Integration, DidNotEnable
from sentry_sdk.tracing import Span
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.utils import capture_internal_exceptions
from typing import TypeVar
def _inner_send_data(*args: P.args, **kwargs: P.kwargs) -> T:
    instance = args[0]
    data = args[2]
    span = instance.connection._sentry_span
    _set_db_data(span, instance.connection)
    if _should_send_default_pii():
        db_params = span._data.get('db.params', [])
        db_params.extend(data)
        span.set_data('db.params', db_params)
    return f(*args, **kwargs)