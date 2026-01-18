from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
from cloudsdk.google.protobuf import duration_pb2  # type: ignore
from cloudsdk.google.protobuf import field_mask_pb2  # type: ignore
from cloudsdk.google.protobuf import timestamp_pb2  # type: ignore
from google.type import date_pb2  # type: ignore
class StartResumableWriteResponse(proto.Message):
    """Response object for ``StartResumableWrite``.

    Attributes:
        upload_id (str):
            The upload_id of the newly started resumable write
            operation. This value should be copied into the
            ``WriteObjectRequest.upload_id`` field.
    """
    upload_id: str = proto.Field(proto.STRING, number=1)