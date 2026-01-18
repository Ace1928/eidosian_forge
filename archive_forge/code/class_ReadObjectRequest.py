from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
from cloudsdk.google.protobuf import duration_pb2  # type: ignore
from cloudsdk.google.protobuf import field_mask_pb2  # type: ignore
from cloudsdk.google.protobuf import timestamp_pb2  # type: ignore
from google.type import date_pb2  # type: ignore
class ReadObjectRequest(proto.Message):
    """Request message for ReadObject.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        bucket (str):
            Required. The name of the bucket containing
            the object to read.
        object_ (str):
            Required. The name of the object to read.
        generation (int):
            If present, selects a specific revision of
            this object (as opposed to the latest version,
            the default).
        read_offset (int):
            The offset for the first byte to return in the read,
            relative to the start of the object.

            A negative ``read_offset`` value will be interpreted as the
            number of bytes back from the end of the object to be
            returned. For example, if an object's length is 15 bytes, a
            ReadObjectRequest with ``read_offset`` = -5 and
            ``read_limit`` = 3 would return bytes 10 through 12 of the
            object. Requesting a negative offset with magnitude larger
            than the size of the object will return the entire object.
        read_limit (int):
            The maximum number of ``data`` bytes the server is allowed
            to return in the sum of all ``Object`` messages. A
            ``read_limit`` of zero indicates that there is no limit, and
            a negative ``read_limit`` will cause an error.

            If the stream returns fewer bytes than allowed by the
            ``read_limit`` and no error occurred, the stream includes
            all data from the ``read_offset`` to the end of the
            resource.
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
        common_object_request_params (googlecloudsdk.generated_clients.gapic_clients.storage_v2.types.CommonObjectRequestParams):
            A set of parameters common to Storage API
            requests concerning an object.
        read_mask (google.protobuf.field_mask_pb2.FieldMask):
            Mask specifying which fields to read. The checksummed_data
            field and its children will always be present. If no mask is
            specified, will default to all fields except metadata.owner
            and metadata.acl.

            -  may be used to mean "all fields".

            This field is a member of `oneof`_ ``_read_mask``.
    """
    bucket: str = proto.Field(proto.STRING, number=1)
    object_: str = proto.Field(proto.STRING, number=2)
    generation: int = proto.Field(proto.INT64, number=3)
    read_offset: int = proto.Field(proto.INT64, number=4)
    read_limit: int = proto.Field(proto.INT64, number=5)
    if_generation_match: int = proto.Field(proto.INT64, number=6, optional=True)
    if_generation_not_match: int = proto.Field(proto.INT64, number=7, optional=True)
    if_metageneration_match: int = proto.Field(proto.INT64, number=8, optional=True)
    if_metageneration_not_match: int = proto.Field(proto.INT64, number=9, optional=True)
    common_object_request_params: 'CommonObjectRequestParams' = proto.Field(proto.MESSAGE, number=10, message='CommonObjectRequestParams')
    read_mask: field_mask_pb2.FieldMask = proto.Field(proto.MESSAGE, number=12, optional=True, message=field_mask_pb2.FieldMask)