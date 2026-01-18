from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
from google.cloud.pubsublite_v1.types import common
from cloudsdk.google.protobuf import field_mask_pb2  # type: ignore
from cloudsdk.google.protobuf import timestamp_pb2  # type: ignore
class CreateSubscriptionRequest(proto.Message):
    """Request for CreateSubscription.

    Attributes:
        parent (str):
            Required. The parent location in which to create the
            subscription. Structured like
            ``projects/{project_number}/locations/{location}``.
        subscription (google.cloud.pubsublite_v1.types.Subscription):
            Required. Configuration of the subscription to create. Its
            ``name`` field is ignored.
        subscription_id (str):
            Required. The ID to use for the subscription, which will
            become the final component of the subscription's name.

            This value is structured like: ``my-sub-name``.
        skip_backlog (bool):
            If true, the newly created subscription will
            only receive messages published after the
            subscription was created. Otherwise, the entire
            message backlog will be received on the
            subscription. Defaults to false.
    """
    parent: str = proto.Field(proto.STRING, number=1)
    subscription: common.Subscription = proto.Field(proto.MESSAGE, number=2, message=common.Subscription)
    subscription_id: str = proto.Field(proto.STRING, number=3)
    skip_backlog: bool = proto.Field(proto.BOOL, number=4)