from __future__ import annotations
from typing import MutableMapping, MutableSequence
from google.protobuf import duration_pb2  # type: ignore
from google.protobuf import timestamp_pb2  # type: ignore
from google.protobuf import wrappers_pb2  # type: ignore
from google.rpc import status_pb2  # type: ignore
import proto  # type: ignore
from google.cloud.speech_v1p1beta1.types import resource
class StreamingRecognizeResponse(proto.Message):
    """``StreamingRecognizeResponse`` is the only message returned to the
    client by ``StreamingRecognize``. A series of zero or more
    ``StreamingRecognizeResponse`` messages are streamed back to the
    client. If there is no recognizable audio, and ``single_utterance``
    is set to false, then no messages are streamed back to the client.

    Here's an example of a series of ``StreamingRecognizeResponse``\\ s
    that might be returned while processing audio:

    1. results { alternatives { transcript: "tube" } stability: 0.01 }

    2. results { alternatives { transcript: "to be a" } stability: 0.01
       }

    3. results { alternatives { transcript: "to be" } stability: 0.9 }
       results { alternatives { transcript: " or not to be" } stability:
       0.01 }

    4. results { alternatives { transcript: "to be or not to be"
       confidence: 0.92 } alternatives { transcript: "to bee or not to
       bee" } is_final: true }

    5. results { alternatives { transcript: " that's" } stability: 0.01
       }

    6. results { alternatives { transcript: " that is" } stability: 0.9
       } results { alternatives { transcript: " the question" }
       stability: 0.01 }

    7. results { alternatives { transcript: " that is the question"
       confidence: 0.98 } alternatives { transcript: " that was the
       question" } is_final: true }

    Notes:

    -  Only two of the above responses #4 and #7 contain final results;
       they are indicated by ``is_final: true``. Concatenating these
       together generates the full transcript: "to be or not to be that
       is the question".

    -  The others contain interim ``results``. #3 and #6 contain two
       interim ``results``: the first portion has a high stability and
       is less likely to change; the second portion has a low stability
       and is very likely to change. A UI designer might choose to show
       only high stability ``results``.

    -  The specific ``stability`` and ``confidence`` values shown above
       are only for illustrative purposes. Actual values may vary.

    -  In each response, only one of these fields will be set:
       ``error``, ``speech_event_type``, or one or more (repeated)
       ``results``.

    Attributes:
        error (google.rpc.status_pb2.Status):
            If set, returns a [google.rpc.Status][google.rpc.Status]
            message that specifies the error for the operation.
        results (MutableSequence[google.cloud.speech_v1p1beta1.types.StreamingRecognitionResult]):
            This repeated list contains zero or more results that
            correspond to consecutive portions of the audio currently
            being processed. It contains zero or one ``is_final=true``
            result (the newly settled portion), followed by zero or more
            ``is_final=false`` results (the interim results).
        speech_event_type (google.cloud.speech_v1p1beta1.types.StreamingRecognizeResponse.SpeechEventType):
            Indicates the type of speech event.
        speech_event_time (google.protobuf.duration_pb2.Duration):
            Time offset between the beginning of the
            audio and event emission.
        total_billed_time (google.protobuf.duration_pb2.Duration):
            When available, billed audio seconds for the
            stream. Set only if this is the last response in
            the stream.
        speech_adaptation_info (google.cloud.speech_v1p1beta1.types.SpeechAdaptationInfo):
            Provides information on adaptation behavior
            in response
        request_id (int):
            The ID associated with the request. This is a
            unique ID specific only to the given request.
    """

    class SpeechEventType(proto.Enum):
        """Indicates the type of speech event.

        Values:
            SPEECH_EVENT_UNSPECIFIED (0):
                No speech event specified.
            END_OF_SINGLE_UTTERANCE (1):
                This event indicates that the server has detected the end of
                the user's speech utterance and expects no additional
                speech. Therefore, the server will not process additional
                audio (although it may subsequently return additional
                results). The client should stop sending additional audio
                data, half-close the gRPC connection, and wait for any
                additional results until the server closes the gRPC
                connection. This event is only sent if ``single_utterance``
                was set to ``true``, and is not used otherwise.
            SPEECH_ACTIVITY_BEGIN (2):
                This event indicates that the server has detected the
                beginning of human voice activity in the stream. This event
                can be returned multiple times if speech starts and stops
                repeatedly throughout the stream. This event is only sent if
                ``voice_activity_events`` is set to true.
            SPEECH_ACTIVITY_END (3):
                This event indicates that the server has detected the end of
                human voice activity in the stream. This event can be
                returned multiple times if speech starts and stops
                repeatedly throughout the stream. This event is only sent if
                ``voice_activity_events`` is set to true.
            SPEECH_ACTIVITY_TIMEOUT (4):
                This event indicates that the user-set
                timeout for speech activity begin or end has
                exceeded. Upon receiving this event, the client
                is expected to send a half close. Further audio
                will not be processed.
        """
        SPEECH_EVENT_UNSPECIFIED = 0
        END_OF_SINGLE_UTTERANCE = 1
        SPEECH_ACTIVITY_BEGIN = 2
        SPEECH_ACTIVITY_END = 3
        SPEECH_ACTIVITY_TIMEOUT = 4
    error: status_pb2.Status = proto.Field(proto.MESSAGE, number=1, message=status_pb2.Status)
    results: MutableSequence['StreamingRecognitionResult'] = proto.RepeatedField(proto.MESSAGE, number=2, message='StreamingRecognitionResult')
    speech_event_type: SpeechEventType = proto.Field(proto.ENUM, number=4, enum=SpeechEventType)
    speech_event_time: duration_pb2.Duration = proto.Field(proto.MESSAGE, number=8, message=duration_pb2.Duration)
    total_billed_time: duration_pb2.Duration = proto.Field(proto.MESSAGE, number=5, message=duration_pb2.Duration)
    speech_adaptation_info: 'SpeechAdaptationInfo' = proto.Field(proto.MESSAGE, number=9, message='SpeechAdaptationInfo')
    request_id: int = proto.Field(proto.INT64, number=10)