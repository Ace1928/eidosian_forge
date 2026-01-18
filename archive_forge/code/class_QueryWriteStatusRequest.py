from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
from cloudsdk.google.protobuf import duration_pb2  # type: ignore
from cloudsdk.google.protobuf import field_mask_pb2  # type: ignore
from cloudsdk.google.protobuf import timestamp_pb2  # type: ignore
from google.type import date_pb2  # type: ignore
class QueryWriteStatusRequest(proto.Message):
    """Request object for ``QueryWriteStatus``.

    Attributes:
        upload_id (str):
            Required. The name of the resume token for
            the object whose write status is being
            requested.
        common_object_request_params (googlecloudsdk.generated_clients.gapic_clients.storage_v2.types.CommonObjectRequestParams):
            A set of parameters common to Storage API
            requests concerning an object.
    """
    upload_id: str = proto.Field(proto.STRING, number=1)
    common_object_request_params: 'CommonObjectRequestParams' = proto.Field(proto.MESSAGE, number=2, message='CommonObjectRequestParams')