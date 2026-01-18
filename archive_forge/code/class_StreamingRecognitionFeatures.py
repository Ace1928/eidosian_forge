from __future__ import annotations
from typing import MutableMapping, MutableSequence
from google.protobuf import duration_pb2  # type: ignore
from google.protobuf import field_mask_pb2  # type: ignore
from google.protobuf import timestamp_pb2  # type: ignore
from google.rpc import status_pb2  # type: ignore
import proto  # type: ignore
class StreamingRecognitionFeatures(proto.Message):
    """Available recognition features specific to streaming
    recognition requests.

    Attributes:
        enable_voice_activity_events (bool):
            If ``true``, responses with voice activity speech events
            will be returned as they are detected.
        interim_results (bool):
            Whether or not to stream interim results to
            the client. If set to true, interim results will
            be streamed to the client. Otherwise, only the
            final response will be streamed back.
        voice_activity_timeout (google.cloud.speech_v2.types.StreamingRecognitionFeatures.VoiceActivityTimeout):
            If set, the server will automatically close the stream after
            the specified duration has elapsed after the last
            VOICE_ACTIVITY speech event has been sent. The field
            ``voice_activity_events`` must also be set to true.
    """

    class VoiceActivityTimeout(proto.Message):
        """Events that a timeout can be set on for voice activity.

        Attributes:
            speech_start_timeout (google.protobuf.duration_pb2.Duration):
                Duration to timeout the stream if no speech
                begins. If this is set and no speech is detected
                in this duration at the start of the stream, the
                server will close the stream.
            speech_end_timeout (google.protobuf.duration_pb2.Duration):
                Duration to timeout the stream after speech
                ends. If this is set and no speech is detected
                in this duration after speech was detected, the
                server will close the stream.
        """
        speech_start_timeout: duration_pb2.Duration = proto.Field(proto.MESSAGE, number=1, message=duration_pb2.Duration)
        speech_end_timeout: duration_pb2.Duration = proto.Field(proto.MESSAGE, number=2, message=duration_pb2.Duration)
    enable_voice_activity_events: bool = proto.Field(proto.BOOL, number=1)
    interim_results: bool = proto.Field(proto.BOOL, number=2)
    voice_activity_timeout: VoiceActivityTimeout = proto.Field(proto.MESSAGE, number=3, message=VoiceActivityTimeout)