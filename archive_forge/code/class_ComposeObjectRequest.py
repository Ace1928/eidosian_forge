from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
from cloudsdk.google.protobuf import duration_pb2  # type: ignore
from cloudsdk.google.protobuf import field_mask_pb2  # type: ignore
from cloudsdk.google.protobuf import timestamp_pb2  # type: ignore
from google.type import date_pb2  # type: ignore
class ComposeObjectRequest(proto.Message):
    """Request message for ComposeObject.

    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        destination (googlecloudsdk.generated_clients.gapic_clients.storage_v2.types.Object):
            Required. Properties of the resulting object.
        source_objects (MutableSequence[googlecloudsdk.generated_clients.gapic_clients.storage_v2.types.ComposeObjectRequest.SourceObject]):
            The list of source objects that will be
            concatenated into a single object.
        destination_predefined_acl (str):
            Apply a predefined set of access controls to
            the destination object. Valid values are
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
        if_metageneration_match (int):
            Makes the operation conditional on whether
            the object's current metageneration matches the
            given value.

            This field is a member of `oneof`_ ``_if_metageneration_match``.
        kms_key (str):
            Resource name of the Cloud KMS key, of the form
            ``projects/my-project/locations/my-location/keyRings/my-kr/cryptoKeys/my-key``,
            that will be used to encrypt the object. Overrides the
            object metadata's ``kms_key_name`` value, if any.
        common_object_request_params (googlecloudsdk.generated_clients.gapic_clients.storage_v2.types.CommonObjectRequestParams):
            A set of parameters common to Storage API
            requests concerning an object.
        object_checksums (googlecloudsdk.generated_clients.gapic_clients.storage_v2.types.ObjectChecksums):
            The checksums of the complete object. This
            will be validated against the combined checksums
            of the component objects.
    """

    class SourceObject(proto.Message):
        """Description of a source object for a composition request.

        Attributes:
            name (str):
                Required. The source object's name. All
                source objects must reside in the same bucket.
            generation (int):
                The generation of this object to use as the
                source.
            object_preconditions (googlecloudsdk.generated_clients.gapic_clients.storage_v2.types.ComposeObjectRequest.SourceObject.ObjectPreconditions):
                Conditions that must be met for this
                operation to execute.
        """

        class ObjectPreconditions(proto.Message):
            """Preconditions for a source object of a composition request.

            .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

            Attributes:
                if_generation_match (int):
                    Only perform the composition if the
                    generation of the source object that would be
                    used matches this value.  If this value and a
                    generation are both specified, they must be the
                    same value or the call will fail.

                    This field is a member of `oneof`_ ``_if_generation_match``.
            """
            if_generation_match: int = proto.Field(proto.INT64, number=1, optional=True)
        name: str = proto.Field(proto.STRING, number=1)
        generation: int = proto.Field(proto.INT64, number=2)
        object_preconditions: 'ComposeObjectRequest.SourceObject.ObjectPreconditions' = proto.Field(proto.MESSAGE, number=3, message='ComposeObjectRequest.SourceObject.ObjectPreconditions')
    destination: 'Object' = proto.Field(proto.MESSAGE, number=1, message='Object')
    source_objects: MutableSequence[SourceObject] = proto.RepeatedField(proto.MESSAGE, number=2, message=SourceObject)
    destination_predefined_acl: str = proto.Field(proto.STRING, number=9)
    if_generation_match: int = proto.Field(proto.INT64, number=4, optional=True)
    if_metageneration_match: int = proto.Field(proto.INT64, number=5, optional=True)
    kms_key: str = proto.Field(proto.STRING, number=6)
    common_object_request_params: 'CommonObjectRequestParams' = proto.Field(proto.MESSAGE, number=7, message='CommonObjectRequestParams')
    object_checksums: 'ObjectChecksums' = proto.Field(proto.MESSAGE, number=10, message='ObjectChecksums')