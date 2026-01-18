from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
from cloudsdk.google.protobuf import duration_pb2  # type: ignore
from cloudsdk.google.protobuf import field_mask_pb2  # type: ignore
from cloudsdk.google.protobuf import timestamp_pb2  # type: ignore
from google.type import date_pb2  # type: ignore
class ListBucketsRequest(proto.Message):
    """Request message for ListBuckets.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        parent (str):
            Required. The project whose buckets we are
            listing.
        page_size (int):
            Maximum number of buckets to return in a single response.
            The service will use this parameter or 1,000 items,
            whichever is smaller. If "acl" is present in the read_mask,
            the service will use this parameter of 200 items, whichever
            is smaller.
        page_token (str):
            A previously-returned page token representing
            part of the larger set of results to view.
        prefix (str):
            Filter results to buckets whose names begin
            with this prefix.
        read_mask (google.protobuf.field_mask_pb2.FieldMask):
            Mask specifying which fields to read from each result. If no
            mask is specified, will default to all fields except
            items.owner, items.acl, and items.default_object_acl.

            -  may be used to mean "all fields".

            This field is a member of `oneof`_ ``_read_mask``.
    """
    parent: str = proto.Field(proto.STRING, number=1)
    page_size: int = proto.Field(proto.INT32, number=2)
    page_token: str = proto.Field(proto.STRING, number=3)
    prefix: str = proto.Field(proto.STRING, number=4)
    read_mask: field_mask_pb2.FieldMask = proto.Field(proto.MESSAGE, number=5, optional=True, message=field_mask_pb2.FieldMask)