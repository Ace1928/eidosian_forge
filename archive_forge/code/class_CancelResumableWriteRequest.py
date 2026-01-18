from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
from cloudsdk.google.protobuf import duration_pb2  # type: ignore
from cloudsdk.google.protobuf import field_mask_pb2  # type: ignore
from cloudsdk.google.protobuf import timestamp_pb2  # type: ignore
from google.type import date_pb2  # type: ignore
class CancelResumableWriteRequest(proto.Message):
    """Message for canceling an in-progress resumable upload. ``upload_id``
    **must** be set.

    Attributes:
        upload_id (str):
            Required. The upload_id of the resumable upload to cancel.
            This should be copied from the ``upload_id`` field of
            ``StartResumableWriteResponse``.
    """
    upload_id: str = proto.Field(proto.STRING, number=1)