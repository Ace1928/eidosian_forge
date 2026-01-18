from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleApiServicecontrolV1TraceSpan(_messages.Message):
    """A span represents a single operation within a trace. Spans can be nested
  to form a trace tree. Often, a trace contains a root span that describes the
  end-to-end latency, and one or more subspans for its sub-operations. A trace
  can also contain multiple root spans, or none at all. Spans do not need to
  be contiguous-there may be gaps or overlaps between spans in a trace.

  Enums:
    SpanKindValueValuesEnum: Distinguishes between spans generated in a
      particular context. For example, two spans with the same name may be
      distinguished using `CLIENT` (caller) and `SERVER` (callee) to identify
      an RPC call.

  Fields:
    attributes: A set of attributes on the span. You can have up to 32
      attributes per span.
    childSpanCount: An optional number of child spans that were generated
      while this span was active. If set, allows implementation to detect
      missing child spans.
    displayName: A description of the span's operation (up to 128 bytes).
      Stackdriver Trace displays the description in the Google Cloud Platform
      Console. For example, the display name can be a qualified method name or
      a file name and a line number where the operation is called. A best
      practice is to use the same display name within an application and at
      the same call point. This makes it easier to correlate spans in
      different traces.
    endTime: The end time of the span. On the client side, this is the time
      kept by the local machine where the span execution ends. On the server
      side, this is the time when the server application handler stops
      running.
    name: The resource name of the span in the following format:
      projects/[PROJECT_ID]/traces/[TRACE_ID]/spans/SPAN_ID is a unique
      identifier for a trace within a project; it is a 32-character
      hexadecimal encoding of a 16-byte array. [SPAN_ID] is a unique
      identifier for a span within a trace; it is a 16-character hexadecimal
      encoding of an 8-byte array.
    parentSpanId: The [SPAN_ID] of this span's parent span. If this is a root
      span, then this field must be empty.
    sameProcessAsParentSpan: (Optional) Set this parameter to indicate whether
      this span is in the same process as its parent. If you do not set this
      parameter, Stackdriver Trace is unable to take advantage of this helpful
      information.
    spanId: The [SPAN_ID] portion of the span's resource name.
    spanKind: Distinguishes between spans generated in a particular context.
      For example, two spans with the same name may be distinguished using
      `CLIENT` (caller) and `SERVER` (callee) to identify an RPC call.
    startTime: The start time of the span. On the client side, this is the
      time kept by the local machine where the span execution starts. On the
      server side, this is the time when the server's application handler
      starts running.
    status: An optional final status for this span.
  """

    class SpanKindValueValuesEnum(_messages.Enum):
        """Distinguishes between spans generated in a particular context. For
    example, two spans with the same name may be distinguished using `CLIENT`
    (caller) and `SERVER` (callee) to identify an RPC call.

    Values:
      SPAN_KIND_UNSPECIFIED: Unspecified. Do NOT use as default.
        Implementations MAY assume SpanKind.INTERNAL to be default.
      INTERNAL: Indicates that the span is used internally. Default value.
      SERVER: Indicates that the span covers server-side handling of an RPC or
        other remote network request.
      CLIENT: Indicates that the span covers the client-side wrapper around an
        RPC or other remote request.
      PRODUCER: Indicates that the span describes producer sending a message
        to a broker. Unlike client and server, there is no direct critical
        path latency relationship between producer and consumer spans (e.g.
        publishing a message to a pubsub service).
      CONSUMER: Indicates that the span describes consumer receiving a message
        from a broker. Unlike client and server, there is no direct critical
        path latency relationship between producer and consumer spans (e.g.
        receiving a message from a pubsub service subscription).
    """
        SPAN_KIND_UNSPECIFIED = 0
        INTERNAL = 1
        SERVER = 2
        CLIENT = 3
        PRODUCER = 4
        CONSUMER = 5
    attributes = _messages.MessageField('GoogleApiServicecontrolV1Attributes', 1)
    childSpanCount = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    displayName = _messages.MessageField('GoogleApiServicecontrolV1TruncatableString', 3)
    endTime = _messages.StringField(4)
    name = _messages.StringField(5)
    parentSpanId = _messages.StringField(6)
    sameProcessAsParentSpan = _messages.BooleanField(7)
    spanId = _messages.StringField(8)
    spanKind = _messages.EnumField('SpanKindValueValuesEnum', 9)
    startTime = _messages.StringField(10)
    status = _messages.MessageField('Status', 11)