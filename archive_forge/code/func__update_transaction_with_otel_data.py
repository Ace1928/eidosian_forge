from time import time
from opentelemetry.context import get_value  # type: ignore
from opentelemetry.sdk.trace import SpanProcessor  # type: ignore
from opentelemetry.semconv.trace import SpanAttributes  # type: ignore
from opentelemetry.trace import (  # type: ignore
from opentelemetry.trace.span import (  # type: ignore
from sentry_sdk._compat import utc_from_timestamp
from sentry_sdk.consts import INSTRUMENTER
from sentry_sdk.hub import Hub
from sentry_sdk.integrations.opentelemetry.consts import (
from sentry_sdk.scope import add_global_event_processor
from sentry_sdk.tracing import Transaction, Span as SentrySpan
from sentry_sdk.utils import Dsn
from sentry_sdk._types import TYPE_CHECKING
from urllib3.util import parse_url as urlparse
def _update_transaction_with_otel_data(self, sentry_span, otel_span):
    http_method = otel_span.attributes.get(SpanAttributes.HTTP_METHOD)
    if http_method:
        status_code = otel_span.attributes.get(SpanAttributes.HTTP_STATUS_CODE)
        if status_code:
            sentry_span.set_http_status(status_code)
        op = 'http'
        if otel_span.kind == SpanKind.SERVER:
            op += '.server'
        elif otel_span.kind == SpanKind.CLIENT:
            op += '.client'
        sentry_span.op = op