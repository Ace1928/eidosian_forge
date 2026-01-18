from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
from cloudsdk.google.protobuf import duration_pb2  # type: ignore
from cloudsdk.google.protobuf import field_mask_pb2  # type: ignore
from cloudsdk.google.protobuf import timestamp_pb2  # type: ignore
from google.type import date_pb2  # type: ignore
class WriteObjectSpec(proto.Message):
    """Describes an attempt to insert an object, possibly over
    multiple requests.


    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        resource (googlecloudsdk.generated_clients.gapic_clients.storage_v2.types.Object):
            Required. Destination object, including its
            name and its metadata.
        predefined_acl (str):
            Apply a predefined set of access controls to
            this object. Valid values are
            "authenticatedRead", "bucketOwnerFullControl",
            "bucketOwnerRead", "private", "projectPrivate",
            or "publicRead".
        if_generation_match (int):
            Makes the operation conditional on whether
            the object's current generation matches the
            given value. Setting to 0 makes the operation
            succeed only if there are no live versions of
            the object.

            This field is a member of `oneof`_ ``_if_generation_match``.
        if_generation_not_match (int):
            Makes the operation conditional on whether
            the object's live generation does not match the
            given value. If no live object exists, the
            precondition fails. Setting to 0 makes the
            operation succeed only if there is a live
            version of the object.

            This field is a member of `oneof`_ ``_if_generation_not_match``.
        if_metageneration_match (int):
            Makes the operation conditional on whether
            the object's current metageneration matches the
            given value.

            This field is a member of `oneof`_ ``_if_metageneration_match``.
        if_metageneration_not_match (int):
            Makes the operation conditional on whether
            the object's current metageneration does not
            match the given value.

            This field is a member of `oneof`_ ``_if_metageneration_not_match``.
        object_size (int):
            The expected final object size being uploaded. If this value
            is set, closing the stream after writing fewer or more than
            ``object_size`` bytes will result in an OUT_OF_RANGE error.

            This situation is considered a client error, and if such an
            error occurs you must start the upload over from scratch,
            this time sending the correct number of bytes.

            This field is a member of `oneof`_ ``_object_size``.
    """
    resource: 'Object' = proto.Field(proto.MESSAGE, number=1, message='Object')
    predefined_acl: str = proto.Field(proto.STRING, number=7)
    if_generation_match: int = proto.Field(proto.INT64, number=3, optional=True)
    if_generation_not_match: int = proto.Field(proto.INT64, number=4, optional=True)
    if_metageneration_match: int = proto.Field(proto.INT64, number=5, optional=True)
    if_metageneration_not_match: int = proto.Field(proto.INT64, number=6, optional=True)
    object_size: int = proto.Field(proto.INT64, number=8, optional=True)