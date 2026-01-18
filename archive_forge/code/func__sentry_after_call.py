from __future__ import absolute_import
from sentry_sdk import Hub
from sentry_sdk.consts import OP, SPANDATA
from sentry_sdk.integrations import Integration, DidNotEnable
from sentry_sdk.tracing import Span
from sentry_sdk._functools import partial
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.utils import capture_internal_exceptions, parse_url, parse_version
def _sentry_after_call(context, parsed, **kwargs):
    span = context.pop('_sentrysdk_span', None)
    if span is None:
        return
    span.__exit__(None, None, None)
    body = parsed.get('Body')
    if not isinstance(body, StreamingBody):
        return
    streaming_span = span.start_child(op=OP.HTTP_CLIENT_STREAM, description=span.description)
    orig_read = body.read
    orig_close = body.close

    def sentry_streaming_body_read(*args, **kwargs):
        try:
            ret = orig_read(*args, **kwargs)
            if not ret:
                streaming_span.finish()
            return ret
        except Exception:
            streaming_span.finish()
            raise
    body.read = sentry_streaming_body_read

    def sentry_streaming_body_close(*args, **kwargs):
        streaming_span.finish()
        orig_close(*args, **kwargs)
    body.close = sentry_streaming_body_close