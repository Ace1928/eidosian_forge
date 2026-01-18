from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
from cloudsdk.google.protobuf import duration_pb2  # type: ignore
from cloudsdk.google.protobuf import field_mask_pb2  # type: ignore
from cloudsdk.google.protobuf import timestamp_pb2  # type: ignore
from google.type import date_pb2  # type: ignore
class DeleteHmacKeyRequest(proto.Message):
    """Request object to delete a given HMAC key.

    Attributes:
        access_id (str):
            Required. The identifying key for the HMAC to
            delete.
        project (str):
            Required. The project that owns the HMAC key,
            in the format of "projects/{projectIdentifier}".
            {projectIdentifier} can be the project ID or
            project number.
    """
    access_id: str = proto.Field(proto.STRING, number=1)
    project: str = proto.Field(proto.STRING, number=2)