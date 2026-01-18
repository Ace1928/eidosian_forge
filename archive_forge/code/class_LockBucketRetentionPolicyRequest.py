from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
from cloudsdk.google.protobuf import duration_pb2  # type: ignore
from cloudsdk.google.protobuf import field_mask_pb2  # type: ignore
from cloudsdk.google.protobuf import timestamp_pb2  # type: ignore
from google.type import date_pb2  # type: ignore
class LockBucketRetentionPolicyRequest(proto.Message):
    """Request message for LockBucketRetentionPolicyRequest.

    Attributes:
        bucket (str):
            Required. Name of a bucket.
        if_metageneration_match (int):
            Required. Makes the operation conditional on
            whether bucket's current metageneration matches
            the given value. Must be positive.
    """
    bucket: str = proto.Field(proto.STRING, number=1)
    if_metageneration_match: int = proto.Field(proto.INT64, number=2)