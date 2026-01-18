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
def _update_span_with_otel_data(self, sentry_span, otel_span):
    """
        Convert OTel span data and update the Sentry span with it.
        This should eventually happen on the server when ingesting the spans.
        """
    for key, val in otel_span.attributes.items():
        sentry_span.set_data(key, val)
    sentry_span.set_data('otel.kind', otel_span.kind)
    op = otel_span.name
    description = otel_span.name
    http_method = otel_span.attributes.get(SpanAttributes.HTTP_METHOD, None)
    db_query = otel_span.attributes.get(SpanAttributes.DB_SYSTEM, None)
    if http_method:
        op = 'http'
        if otel_span.kind == SpanKind.SERVER:
            op += '.server'
        elif otel_span.kind == SpanKind.CLIENT:
            op += '.client'
        description = http_method
        peer_name = otel_span.attributes.get(SpanAttributes.NET_PEER_NAME, None)
        if peer_name:
            description += ' {}'.format(peer_name)
        target = otel_span.attributes.get(SpanAttributes.HTTP_TARGET, None)
        if target:
            description += ' {}'.format(target)
        if not peer_name and (not target):
            url = otel_span.attributes.get(SpanAttributes.HTTP_URL, None)
            if url:
                parsed_url = urlparse(url)
                url = '{}://{}{}'.format(parsed_url.scheme, parsed_url.netloc, parsed_url.path)
                description += ' {}'.format(url)
        status_code = otel_span.attributes.get(SpanAttributes.HTTP_STATUS_CODE, None)
        if status_code:
            sentry_span.set_http_status(status_code)
    elif db_query:
        op = 'db'
        statement = otel_span.attributes.get(SpanAttributes.DB_STATEMENT, None)
        if statement:
            description = statement
    sentry_span.op = op
    sentry_span.description = description