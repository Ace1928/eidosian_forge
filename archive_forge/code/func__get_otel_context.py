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
def _get_otel_context(self, otel_span):
    """
        Returns the OTel context for Sentry.
        See: https://develop.sentry.dev/sdk/performance/opentelemetry/#step-5-add-opentelemetry-context
        """
    ctx = {}
    if otel_span.attributes:
        ctx['attributes'] = dict(otel_span.attributes)
    if otel_span.resource.attributes:
        ctx['resource'] = dict(otel_span.resource.attributes)
    return ctx