from typing import TYPE_CHECKING
from pydantic import BaseModel  # type: ignore
from sentry_sdk.consts import OP
from sentry_sdk.hub import Hub, _should_send_default_pii
from sentry_sdk.integrations import DidNotEnable, Integration
from sentry_sdk.integrations.asgi import SentryAsgiMiddleware
from sentry_sdk.tracing import SOURCE_FOR_STYLE, TRANSACTION_SOURCE_ROUTE
from sentry_sdk.utils import event_from_exception, transaction_from_function
class SentryStarliteASGIMiddleware(SentryAsgiMiddleware):

    def __init__(self, app: 'ASGIApp'):
        super().__init__(app=app, unsafe_context_data=False, transaction_style='endpoint', mechanism_type='asgi')