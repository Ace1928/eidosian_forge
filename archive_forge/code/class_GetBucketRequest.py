from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
from cloudsdk.google.protobuf import duration_pb2  # type: ignore
from cloudsdk.google.protobuf import field_mask_pb2  # type: ignore
from cloudsdk.google.protobuf import timestamp_pb2  # type: ignore
from google.type import date_pb2  # type: ignore
class GetBucketRequest(proto.Message):
    """Request message for GetBucket.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        name (str):
            Required. Name of a bucket.
        if_metageneration_match (int):
            If set, and if the bucket's current
            metageneration does not match the specified
            value, the request will return an error.

            This field is a member of `oneof`_ ``_if_metageneration_match``.
        if_metageneration_not_match (int):
            If set, and if the bucket's current
            metageneration matches the specified value, the
            request will return an error.

            This field is a member of `oneof`_ ``_if_metageneration_not_match``.
        read_mask (google.protobuf.field_mask_pb2.FieldMask):
            Mask specifying which fields to read. A "*" field may be
            used to indicate all fields. If no mask is specified, will
            default to all fields.

            This field is a member of `oneof`_ ``_read_mask``.
    """
    name: str = proto.Field(proto.STRING, number=1)
    if_metageneration_match: int = proto.Field(proto.INT64, number=2, optional=True)
    if_metageneration_not_match: int = proto.Field(proto.INT64, number=3, optional=True)
    read_mask: field_mask_pb2.FieldMask = proto.Field(proto.MESSAGE, number=5, optional=True, message=field_mask_pb2.FieldMask)