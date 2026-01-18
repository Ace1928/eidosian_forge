from __future__ import annotations
from typing import MutableMapping, MutableSequence
from google.protobuf import duration_pb2  # type: ignore
from google.protobuf import field_mask_pb2  # type: ignore
from google.protobuf import timestamp_pb2  # type: ignore
from google.rpc import status_pb2  # type: ignore
import proto  # type: ignore
class GetRecognizerRequest(proto.Message):
    """Request message for the
    [GetRecognizer][google.cloud.speech.v2.Speech.GetRecognizer] method.

    Attributes:
        name (str):
            Required. The name of the Recognizer to retrieve. The
            expected format is
            ``projects/{project}/locations/{location}/recognizers/{recognizer}``.
    """
    name: str = proto.Field(proto.STRING, number=1)