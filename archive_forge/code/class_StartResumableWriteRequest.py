from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
from cloudsdk.google.protobuf import duration_pb2  # type: ignore
from cloudsdk.google.protobuf import field_mask_pb2  # type: ignore
from cloudsdk.google.protobuf import timestamp_pb2  # type: ignore
from google.type import date_pb2  # type: ignore
class StartResumableWriteRequest(proto.Message):
    """Request message StartResumableWrite.

    Attributes:
        write_object_spec (googlecloudsdk.generated_clients.gapic_clients.storage_v2.types.WriteObjectSpec):
            Required. The destination bucket, object, and
            metadata, as well as any preconditions.
        common_object_request_params (googlecloudsdk.generated_clients.gapic_clients.storage_v2.types.CommonObjectRequestParams):
            A set of parameters common to Storage API
            requests concerning an object.
        object_checksums (googlecloudsdk.generated_clients.gapic_clients.storage_v2.types.ObjectChecksums):
            The checksums of the complete object. This will be used to
            validate the uploaded object. For each upload,
            object_checksums can be provided with either
            StartResumableWriteRequest or the WriteObjectRequest with
            finish_write set to ``true``.
    """
    write_object_spec: 'WriteObjectSpec' = proto.Field(proto.MESSAGE, number=1, message='WriteObjectSpec')
    common_object_request_params: 'CommonObjectRequestParams' = proto.Field(proto.MESSAGE, number=3, message='CommonObjectRequestParams')
    object_checksums: 'ObjectChecksums' = proto.Field(proto.MESSAGE, number=5, message='ObjectChecksums')