from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
from cloudsdk.google.protobuf import duration_pb2  # type: ignore
from cloudsdk.google.protobuf import field_mask_pb2  # type: ignore
from cloudsdk.google.protobuf import timestamp_pb2  # type: ignore
from google.type import date_pb2  # type: ignore
class CustomPlacementConfig(proto.Message):
    """Configuration for Custom Dual Regions. It should specify precisely
        two eligible regions within the same Multiregion. More information
        on regions may be found
        [https://cloud.google.com/storage/docs/locations][here].

        Attributes:
            data_locations (MutableSequence[str]):
                List of locations to use for data placement.
        """
    data_locations: MutableSequence[str] = proto.RepeatedField(proto.STRING, number=1)