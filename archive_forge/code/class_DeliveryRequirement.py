from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
from cloudsdk.google.protobuf import duration_pb2  # type: ignore
from cloudsdk.google.protobuf import timestamp_pb2  # type: ignore
class DeliveryRequirement(proto.Enum):
    """When this subscription should send messages to subscribers relative
            to messages persistence in storage. For details, see `Creating Lite
            subscriptions <https://cloud.google.com/pubsub/lite/docs/subscriptions#creating_lite_subscriptions>`__.

            Values:
                DELIVERY_REQUIREMENT_UNSPECIFIED (0):
                    Default value. This value is unused.
                DELIVER_IMMEDIATELY (1):
                    The server does not wait for a published
                    message to be successfully written to storage
                    before delivering it to subscribers.
                DELIVER_AFTER_STORED (2):
                    The server will not deliver a published
                    message to subscribers until the message has
                    been successfully written to storage. This will
                    result in higher end-to-end latency, but
                    consistent delivery.
            """
    DELIVERY_REQUIREMENT_UNSPECIFIED = 0
    DELIVER_IMMEDIATELY = 1
    DELIVER_AFTER_STORED = 2