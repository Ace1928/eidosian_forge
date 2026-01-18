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
def _is_sentry_span(self, hub, otel_span):
    """
        Break infinite loop:
        HTTP requests to Sentry are caught by OTel and send again to Sentry.
        """
    otel_span_url = otel_span.attributes.get(SpanAttributes.HTTP_URL, None)
    dsn_url = hub.client and Dsn(hub.client.dsn or '').netloc
    if otel_span_url and dsn_url in otel_span_url:
        return True
    return False