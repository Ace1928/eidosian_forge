from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
from google.cloud.pubsublite_v1.types import common
from cloudsdk.google.protobuf import field_mask_pb2  # type: ignore
from cloudsdk.google.protobuf import timestamp_pb2  # type: ignore
class DeleteTopicRequest(proto.Message):
    """Request for DeleteTopic.

    Attributes:
        name (str):
            Required. The name of the topic to delete.
    """
    name: str = proto.Field(proto.STRING, number=1)