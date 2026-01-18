from __future__ import annotations
from typing import MutableMapping, MutableSequence
from google.protobuf import duration_pb2  # type: ignore
from google.protobuf import field_mask_pb2  # type: ignore
from google.protobuf import timestamp_pb2  # type: ignore
from google.rpc import status_pb2  # type: ignore
import proto  # type: ignore
class GetConfigRequest(proto.Message):
    """Request message for the
    [GetConfig][google.cloud.speech.v2.Speech.GetConfig] method.

    Attributes:
        name (str):
            Required. The name of the config to retrieve. There is
            exactly one config resource per project per location. The
            expected format is
            ``projects/{project}/locations/{location}/config``.
    """
    name: str = proto.Field(proto.STRING, number=1)