from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
from cloudsdk.google.protobuf import duration_pb2  # type: ignore
from cloudsdk.google.protobuf import field_mask_pb2  # type: ignore
from cloudsdk.google.protobuf import timestamp_pb2  # type: ignore
from google.type import date_pb2  # type: ignore
class Versioning(proto.Message):
    """Properties of a bucket related to versioning.
        For more on Cloud Storage versioning, see
        https://cloud.google.com/storage/docs/object-versioning.

        Attributes:
            enabled (bool):
                While set to true, versioning is fully
                enabled for this bucket.
        """
    enabled: bool = proto.Field(proto.BOOL, number=1)