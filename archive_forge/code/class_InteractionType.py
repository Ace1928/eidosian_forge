from __future__ import annotations
from typing import MutableMapping, MutableSequence
from google.protobuf import duration_pb2  # type: ignore
from google.protobuf import timestamp_pb2  # type: ignore
from google.protobuf import wrappers_pb2  # type: ignore
from google.rpc import status_pb2  # type: ignore
import proto  # type: ignore
from google.cloud.speech_v1p1beta1.types import resource
class InteractionType(proto.Enum):
    """Use case categories that the audio recognition request can be
        described by.

        Values:
            INTERACTION_TYPE_UNSPECIFIED (0):
                Use case is either unknown or is something
                other than one of the other values below.
            DISCUSSION (1):
                Multiple people in a conversation or discussion. For example
                in a meeting with two or more people actively participating.
                Typically all the primary people speaking would be in the
                same room (if not, see PHONE_CALL)
            PRESENTATION (2):
                One or more persons lecturing or presenting
                to others, mostly uninterrupted.
            PHONE_CALL (3):
                A phone-call or video-conference in which two
                or more people, who are not in the same room,
                are actively participating.
            VOICEMAIL (4):
                A recorded message intended for another
                person to listen to.
            PROFESSIONALLY_PRODUCED (5):
                Professionally produced audio (eg. TV Show,
                Podcast).
            VOICE_SEARCH (6):
                Transcribe spoken questions and queries into
                text.
            VOICE_COMMAND (7):
                Transcribe voice commands, such as for
                controlling a device.
            DICTATION (8):
                Transcribe speech to text to create a written
                document, such as a text-message, email or
                report.
        """
    INTERACTION_TYPE_UNSPECIFIED = 0
    DISCUSSION = 1
    PRESENTATION = 2
    PHONE_CALL = 3
    VOICEMAIL = 4
    PROFESSIONALLY_PRODUCED = 5
    VOICE_SEARCH = 6
    VOICE_COMMAND = 7
    DICTATION = 8