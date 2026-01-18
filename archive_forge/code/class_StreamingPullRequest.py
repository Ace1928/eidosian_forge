from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
from cloudsdk.google.protobuf import duration_pb2  # type: ignore
from cloudsdk.google.protobuf import field_mask_pb2  # type: ignore
from cloudsdk.google.protobuf import timestamp_pb2  # type: ignore
from google.pubsub_v1.types import schema as gp_schema
class StreamingPullRequest(proto.Message):
    """Request for the ``StreamingPull`` streaming RPC method. This request
    is used to establish the initial stream as well as to stream
    acknowledgements and ack deadline modifications from the client to
    the server.

    Attributes:
        subscription (str):
            Required. The subscription for which to initialize the new
            stream. This must be provided in the first request on the
            stream, and must not be set in subsequent requests from
            client to server. Format is
            ``projects/{project}/subscriptions/{sub}``.
        ack_ids (MutableSequence[str]):
            List of acknowledgement IDs for acknowledging previously
            received messages (received on this stream or a different
            stream). If an ack ID has expired, the corresponding message
            may be redelivered later. Acknowledging a message more than
            once will not result in an error. If the acknowledgement ID
            is malformed, the stream will be aborted with status
            ``INVALID_ARGUMENT``.
        modify_deadline_seconds (MutableSequence[int]):
            The list of new ack deadlines for the IDs listed in
            ``modify_deadline_ack_ids``. The size of this list must be
            the same as the size of ``modify_deadline_ack_ids``. If it
            differs the stream will be aborted with
            ``INVALID_ARGUMENT``. Each element in this list is applied
            to the element in the same position in
            ``modify_deadline_ack_ids``. The new ack deadline is with
            respect to the time this request was sent to the Pub/Sub
            system. Must be >= 0. For example, if the value is 10, the
            new ack deadline will expire 10 seconds after this request
            is received. If the value is 0, the message is immediately
            made available for another streaming or non-streaming pull
            request. If the value is < 0 (an error), the stream will be
            aborted with status ``INVALID_ARGUMENT``.
        modify_deadline_ack_ids (MutableSequence[str]):
            List of acknowledgement IDs whose deadline will be modified
            based on the corresponding element in
            ``modify_deadline_seconds``. This field can be used to
            indicate that more time is needed to process a message by
            the subscriber, or to make the message available for
            redelivery if the processing was interrupted.
        stream_ack_deadline_seconds (int):
            Required. The ack deadline to use for the
            stream. This must be provided in the first
            request on the stream, but it can also be
            updated on subsequent requests from client to
            server. The minimum deadline you can specify is
            10 seconds. The maximum deadline you can specify
            is 600 seconds (10 minutes).
        client_id (str):
            A unique identifier that is used to distinguish client
            instances from each other. Only needs to be provided on the
            initial request. When a stream disconnects and reconnects
            for the same stream, the client_id should be set to the same
            value so that state associated with the old stream can be
            transferred to the new stream. The same client_id should not
            be used for different client instances.
        max_outstanding_messages (int):
            Flow control settings for the maximum number of outstanding
            messages. When there are ``max_outstanding_messages`` or
            more currently sent to the streaming pull client that have
            not yet been acked or nacked, the server stops sending more
            messages. The sending of messages resumes once the number of
            outstanding messages is less than this value. If the value
            is <= 0, there is no limit to the number of outstanding
            messages. This property can only be set on the initial
            StreamingPullRequest. If it is set on a subsequent request,
            the stream will be aborted with status ``INVALID_ARGUMENT``.
        max_outstanding_bytes (int):
            Flow control settings for the maximum number of outstanding
            bytes. When there are ``max_outstanding_bytes`` or more
            worth of messages currently sent to the streaming pull
            client that have not yet been acked or nacked, the server
            will stop sending more messages. The sending of messages
            resumes once the number of outstanding bytes is less than
            this value. If the value is <= 0, there is no limit to the
            number of outstanding bytes. This property can only be set
            on the initial StreamingPullRequest. If it is set on a
            subsequent request, the stream will be aborted with status
            ``INVALID_ARGUMENT``.
    """
    subscription: str = proto.Field(proto.STRING, number=1)
    ack_ids: MutableSequence[str] = proto.RepeatedField(proto.STRING, number=2)
    modify_deadline_seconds: MutableSequence[int] = proto.RepeatedField(proto.INT32, number=3)
    modify_deadline_ack_ids: MutableSequence[str] = proto.RepeatedField(proto.STRING, number=4)
    stream_ack_deadline_seconds: int = proto.Field(proto.INT32, number=5)
    client_id: str = proto.Field(proto.STRING, number=6)
    max_outstanding_messages: int = proto.Field(proto.INT64, number=7)
    max_outstanding_bytes: int = proto.Field(proto.INT64, number=8)