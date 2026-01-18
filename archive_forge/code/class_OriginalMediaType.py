from __future__ import annotations
from typing import MutableMapping, MutableSequence
from google.protobuf import duration_pb2  # type: ignore
from google.protobuf import timestamp_pb2  # type: ignore
from google.protobuf import wrappers_pb2  # type: ignore
from google.rpc import status_pb2  # type: ignore
import proto  # type: ignore
from google.cloud.speech_v1p1beta1.types import resource
class OriginalMediaType(proto.Enum):
    """The original media the speech was recorded on.

        Values:
            ORIGINAL_MEDIA_TYPE_UNSPECIFIED (0):
                Unknown original media type.
            AUDIO (1):
                The speech data is an audio recording.
            VIDEO (2):
                The speech data originally recorded on a
                video.
        """
    ORIGINAL_MEDIA_TYPE_UNSPECIFIED = 0
    AUDIO = 1
    VIDEO = 2