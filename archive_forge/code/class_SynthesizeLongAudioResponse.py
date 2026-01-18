from __future__ import annotations
from typing import MutableMapping, MutableSequence
from google.protobuf import timestamp_pb2  # type: ignore
import proto  # type: ignore
from google.cloud.texttospeech_v1.types import cloud_tts
class SynthesizeLongAudioResponse(proto.Message):
    """The message returned to the client by the ``SynthesizeLongAudio``
    method.

    """