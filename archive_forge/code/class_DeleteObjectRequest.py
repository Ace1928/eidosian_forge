from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
from cloudsdk.google.protobuf import duration_pb2  # type: ignore
from cloudsdk.google.protobuf import field_mask_pb2  # type: ignore
from cloudsdk.google.protobuf import timestamp_pb2  # type: ignore
from google.type import date_pb2  # type: ignore
class DeleteObjectRequest(proto.Message):
    """Message for deleting an object. ``bucket`` and ``object`` **must**
    be set.


    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        bucket (str):
            Required. Name of the bucket in which the
            object resides.
        object_ (str):
            Required. The name of the finalized object to delete. Note:
            If you want to delete an unfinalized resumable upload please
            use ``CancelResumableWrite``.
        generation (int):
            If present, permanently deletes a specific
            revision of this object (as opposed to the
            latest version, the default).
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
    """
    bucket: str = proto.Field(proto.STRING, number=1)
    object_: str = proto.Field(proto.STRING, number=2)
    generation: int = proto.Field(proto.INT64, number=4)
    if_generation_match: int = proto.Field(proto.INT64, number=5, optional=True)
    if_generation_not_match: int = proto.Field(proto.INT64, number=6, optional=True)
    if_metageneration_match: int = proto.Field(proto.INT64, number=7, optional=True)
    if_metageneration_not_match: int = proto.Field(proto.INT64, number=8, optional=True)
    common_object_request_params: 'CommonObjectRequestParams' = proto.Field(proto.MESSAGE, number=10, message='CommonObjectRequestParams')