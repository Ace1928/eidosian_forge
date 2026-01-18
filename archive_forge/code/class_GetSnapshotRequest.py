from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
from cloudsdk.google.protobuf import duration_pb2  # type: ignore
from cloudsdk.google.protobuf import field_mask_pb2  # type: ignore
from cloudsdk.google.protobuf import timestamp_pb2  # type: ignore
from google.pubsub_v1.types import schema as gp_schema
class GetSnapshotRequest(proto.Message):
    """Request for the GetSnapshot method.

    Attributes:
        snapshot (str):
            Required. The name of the snapshot to get. Format is
            ``projects/{project}/snapshots/{snap}``.
    """
    snapshot: str = proto.Field(proto.STRING, number=1)