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
def _prune_old_spans(self):
    """
        Prune spans that have been open for too long.
        """
    current_time_minutes = int(time() / 60)
    for span_start_minutes in list(self.open_spans.keys()):
        if self.open_spans[span_start_minutes] == set():
            self.open_spans.pop(span_start_minutes)
        elif current_time_minutes - span_start_minutes > SPAN_MAX_TIME_OPEN_MINUTES:
            for span_id in self.open_spans.pop(span_start_minutes):
                self.otel_span_map.pop(span_id, None)