from __future__ import annotations
from typing import MutableMapping, MutableSequence
from google.protobuf import duration_pb2  # type: ignore
from google.protobuf import timestamp_pb2  # type: ignore
from google.protobuf import wrappers_pb2  # type: ignore
from google.rpc import status_pb2  # type: ignore
import proto  # type: ignore
from google.cloud.speech_v1p1beta1.types import resource
Indicates the type of speech event.

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
        