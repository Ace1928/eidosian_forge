from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
from cloudsdk.google.protobuf import duration_pb2  # type: ignore
from cloudsdk.google.protobuf import field_mask_pb2  # type: ignore
from cloudsdk.google.protobuf import timestamp_pb2  # type: ignore
from google.type import date_pb2  # type: ignore
class HierarchicalNamespace(proto.Message):
    """Configuration for a bucket's hierarchical namespace feature.

        Attributes:
            enabled (bool):
                Optional. Enables the hierarchical namespace
                feature.
        """
    enabled: bool = proto.Field(proto.BOOL, number=1)