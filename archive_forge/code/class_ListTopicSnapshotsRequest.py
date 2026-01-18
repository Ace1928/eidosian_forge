from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
from cloudsdk.google.protobuf import duration_pb2  # type: ignore
from cloudsdk.google.protobuf import field_mask_pb2  # type: ignore
from cloudsdk.google.protobuf import timestamp_pb2  # type: ignore
from google.pubsub_v1.types import schema as gp_schema
class ListTopicSnapshotsRequest(proto.Message):
    """Request for the ``ListTopicSnapshots`` method.

    Attributes:
        topic (str):
            Required. The name of the topic that snapshots are attached
            to. Format is ``projects/{project}/topics/{topic}``.
        page_size (int):
            Maximum number of snapshot names to return.
        page_token (str):
            The value returned by the last
            ``ListTopicSnapshotsResponse``; indicates that this is a
            continuation of a prior ``ListTopicSnapshots`` call, and
            that the system should return the next page of data.
    """
    topic: str = proto.Field(proto.STRING, number=1)
    page_size: int = proto.Field(proto.INT32, number=2)
    page_token: str = proto.Field(proto.STRING, number=3)