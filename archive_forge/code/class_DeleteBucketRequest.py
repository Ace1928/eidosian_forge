from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
from cloudsdk.google.protobuf import duration_pb2  # type: ignore
from cloudsdk.google.protobuf import field_mask_pb2  # type: ignore
from cloudsdk.google.protobuf import timestamp_pb2  # type: ignore
from google.type import date_pb2  # type: ignore
class DeleteBucketRequest(proto.Message):
    """Request message for DeleteBucket.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        name (str):
            Required. Name of a bucket to delete.
        if_metageneration_match (int):
            If set, only deletes the bucket if its
            metageneration matches this value.

            This field is a member of `oneof`_ ``_if_metageneration_match``.
        if_metageneration_not_match (int):
            If set, only deletes the bucket if its
            metageneration does not match this value.

            This field is a member of `oneof`_ ``_if_metageneration_not_match``.
    """
    name: str = proto.Field(proto.STRING, number=1)
    if_metageneration_match: int = proto.Field(proto.INT64, number=2, optional=True)
    if_metageneration_not_match: int = proto.Field(proto.INT64, number=3, optional=True)