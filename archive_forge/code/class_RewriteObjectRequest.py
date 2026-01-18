from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
from cloudsdk.google.protobuf import duration_pb2  # type: ignore
from cloudsdk.google.protobuf import field_mask_pb2  # type: ignore
from cloudsdk.google.protobuf import timestamp_pb2  # type: ignore
from google.type import date_pb2  # type: ignore
class RewriteObjectRequest(proto.Message):
    """Request message for RewriteObject. If the source object is encrypted
    using a Customer-Supplied Encryption Key the key information must be
    provided in the copy_source_encryption_algorithm,
    copy_source_encryption_key_bytes, and
    copy_source_encryption_key_sha256_bytes fields. If the destination
    object should be encrypted the keying information should be provided
    in the encryption_algorithm, encryption_key_bytes, and
    encryption_key_sha256_bytes fields of the
    common_object_request_params.customer_encryption field.


    .. _oneof: https://proto-plus-python.readthedocs.io/en/stable/fields.html#oneofs-mutually-exclusive-fields

    Attributes:
        destination_name (str):
            Required. Immutable. The name of the destination object. See
            the `Naming
            Guidelines <https://cloud.google.com/storage/docs/objects#naming>`__.
            Example: ``test.txt`` The ``name`` field by itself does not
            uniquely identify a Cloud Storage object. A Cloud Storage
            object is uniquely identified by the tuple of (bucket,
            object, generation).
        destination_bucket (str):
            Required. Immutable. The name of the bucket
            containing the destination object.
        destination_kms_key (str):
            The name of the Cloud KMS key that will be
            used to encrypt the destination object. The
            Cloud KMS key must be located in same location
            as the object. If the parameter is not
            specified, the request uses the destination
            bucket's default encryption key, if any, or else
            the Google-managed encryption key.
        destination (googlecloudsdk.generated_clients.gapic_clients.storage_v2.types.Object):
            Properties of the destination, post-rewrite object. The
            ``name``, ``bucket`` and ``kms_key`` fields must not be
            populated (these values are specified in the
            ``destination_name``, ``destination_bucket``, and
            ``destination_kms_key`` fields). If ``destination`` is
            present it will be used to construct the destination
            object's metadata; otherwise the destination object's
            metadata will be copied from the source object.
        source_bucket (str):
            Required. Name of the bucket in which to find
            the source object.
        source_object (str):
            Required. Name of the source object.
        source_generation (int):
            If present, selects a specific revision of
            the source object (as opposed to the latest
            version, the default).
        rewrite_token (str):
            Include this field (from the previous rewrite
            response) on each rewrite request after the
            first one, until the rewrite response 'done'
            flag is true. Calls that provide a rewriteToken
            can omit all other request fields, but if
            included those fields must match the values
            provided in the first rewrite request.
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
            the destination object's current metageneration
            matches the given value.

            This field is a member of `oneof`_ ``_if_metageneration_match``.
        if_metageneration_not_match (int):
            Makes the operation conditional on whether
            the destination object's current metageneration
            does not match the given value.

            This field is a member of `oneof`_ ``_if_metageneration_not_match``.
        if_source_generation_match (int):
            Makes the operation conditional on whether
            the source object's live generation matches the
            given value.

            This field is a member of `oneof`_ ``_if_source_generation_match``.
        if_source_generation_not_match (int):
            Makes the operation conditional on whether
            the source object's live generation does not
            match the given value.

            This field is a member of `oneof`_ ``_if_source_generation_not_match``.
        if_source_metageneration_match (int):
            Makes the operation conditional on whether
            the source object's current metageneration
            matches the given value.

            This field is a member of `oneof`_ ``_if_source_metageneration_match``.
        if_source_metageneration_not_match (int):
            Makes the operation conditional on whether
            the source object's current metageneration does
            not match the given value.

            This field is a member of `oneof`_ ``_if_source_metageneration_not_match``.
        max_bytes_rewritten_per_call (int):
            The maximum number of bytes that will be rewritten per
            rewrite request. Most callers shouldn't need to specify this
            parameter - it is primarily in place to support testing. If
            specified the value must be an integral multiple of 1 MiB
            (1048576). Also, this only applies to requests where the
            source and destination span locations and/or storage
            classes. Finally, this value must not change across rewrite
            calls else you'll get an error that the ``rewriteToken`` is
            invalid.
        copy_source_encryption_algorithm (str):
            The algorithm used to encrypt the source
            object, if any. Used if the source object was
            encrypted with a Customer-Supplied Encryption
            Key.
        copy_source_encryption_key_bytes (bytes):
            The raw bytes (not base64-encoded) AES-256
            encryption key used to encrypt the source
            object, if it was encrypted with a
            Customer-Supplied Encryption Key.
        copy_source_encryption_key_sha256_bytes (bytes):
            The raw bytes (not base64-encoded) SHA256
            hash of the encryption key used to encrypt the
            source object, if it was encrypted with a
            Customer-Supplied Encryption Key.
        common_object_request_params (googlecloudsdk.generated_clients.gapic_clients.storage_v2.types.CommonObjectRequestParams):
            A set of parameters common to Storage API
            requests concerning an object.
        object_checksums (googlecloudsdk.generated_clients.gapic_clients.storage_v2.types.ObjectChecksums):
            The checksums of the complete object. This
            will be used to validate the destination object
            after rewriting.
    """
    destination_name: str = proto.Field(proto.STRING, number=24)
    destination_bucket: str = proto.Field(proto.STRING, number=25)
    destination_kms_key: str = proto.Field(proto.STRING, number=27)
    destination: 'Object' = proto.Field(proto.MESSAGE, number=1, message='Object')
    source_bucket: str = proto.Field(proto.STRING, number=2)
    source_object: str = proto.Field(proto.STRING, number=3)
    source_generation: int = proto.Field(proto.INT64, number=4)
    rewrite_token: str = proto.Field(proto.STRING, number=5)
    destination_predefined_acl: str = proto.Field(proto.STRING, number=28)
    if_generation_match: int = proto.Field(proto.INT64, number=7, optional=True)
    if_generation_not_match: int = proto.Field(proto.INT64, number=8, optional=True)
    if_metageneration_match: int = proto.Field(proto.INT64, number=9, optional=True)
    if_metageneration_not_match: int = proto.Field(proto.INT64, number=10, optional=True)
    if_source_generation_match: int = proto.Field(proto.INT64, number=11, optional=True)
    if_source_generation_not_match: int = proto.Field(proto.INT64, number=12, optional=True)
    if_source_metageneration_match: int = proto.Field(proto.INT64, number=13, optional=True)
    if_source_metageneration_not_match: int = proto.Field(proto.INT64, number=14, optional=True)
    max_bytes_rewritten_per_call: int = proto.Field(proto.INT64, number=15)
    copy_source_encryption_algorithm: str = proto.Field(proto.STRING, number=16)
    copy_source_encryption_key_bytes: bytes = proto.Field(proto.BYTES, number=21)
    copy_source_encryption_key_sha256_bytes: bytes = proto.Field(proto.BYTES, number=22)
    common_object_request_params: 'CommonObjectRequestParams' = proto.Field(proto.MESSAGE, number=19, message='CommonObjectRequestParams')
    object_checksums: 'ObjectChecksums' = proto.Field(proto.MESSAGE, number=29, message='ObjectChecksums')