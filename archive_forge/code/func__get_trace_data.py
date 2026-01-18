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
def _get_trace_data(self, otel_span, parent_context):
    """
        Extracts tracing information from one OTel span and its parent OTel context.
        """
    trace_data = {}
    span_context = otel_span.get_span_context()
    span_id = format_span_id(span_context.span_id)
    trace_data['span_id'] = span_id
    trace_id = format_trace_id(span_context.trace_id)
    trace_data['trace_id'] = trace_id
    parent_span_id = format_span_id(otel_span.parent.span_id) if otel_span.parent else None
    trace_data['parent_span_id'] = parent_span_id
    sentry_trace_data = get_value(SENTRY_TRACE_KEY, parent_context)
    trace_data['parent_sampled'] = sentry_trace_data['parent_sampled'] if sentry_trace_data else None
    baggage = get_value(SENTRY_BAGGAGE_KEY, parent_context)
    trace_data['baggage'] = baggage
    return trace_data