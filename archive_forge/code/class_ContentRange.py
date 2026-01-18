from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
from cloudsdk.google.protobuf import duration_pb2  # type: ignore
from cloudsdk.google.protobuf import field_mask_pb2  # type: ignore
from cloudsdk.google.protobuf import timestamp_pb2  # type: ignore
from google.type import date_pb2  # type: ignore
class ContentRange(proto.Message):
    """Specifies a requested range of bytes to download.

    Attributes:
        start (int):
            The starting offset of the object data. This
            value is inclusive.
        end (int):
            The ending offset of the object data. This
            value is exclusive.
        complete_length (int):
            The complete length of the object data.
    """
    start: int = proto.Field(proto.INT64, number=1)
    end: int = proto.Field(proto.INT64, number=2)
    complete_length: int = proto.Field(proto.INT64, number=3)