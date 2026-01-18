from __future__ import annotations
from typing import MutableMapping, MutableSequence
from google.protobuf import duration_pb2  # type: ignore
from google.protobuf import timestamp_pb2  # type: ignore
from google.protobuf import wrappers_pb2  # type: ignore
from google.rpc import status_pb2  # type: ignore
import proto  # type: ignore
from google.cloud.speech_v1p1beta1.types import resource
class VoiceActivityTimeout(proto.Message):
    """Events that a timeout can be set on for voice activity.

        Attributes:
            speech_start_timeout (google.protobuf.duration_pb2.Duration):
                Duration to timeout the stream if no speech
                begins.
            speech_end_timeout (google.protobuf.duration_pb2.Duration):
                Duration to timeout the stream after speech
                ends.
        """
    speech_start_timeout: duration_pb2.Duration = proto.Field(proto.MESSAGE, number=1, message=duration_pb2.Duration)
    speech_end_timeout: duration_pb2.Duration = proto.Field(proto.MESSAGE, number=2, message=duration_pb2.Duration)