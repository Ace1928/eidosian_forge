from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
class SynthesizeSpeechResponse(proto.Message):
    """The message returned to the client by the ``SynthesizeSpeech``
    method.

    Attributes:
        audio_content (bytes):
            The audio data bytes encoded as specified in the request,
            including the header for encodings that are wrapped in
            containers (e.g. MP3, OGG_OPUS). For LINEAR16 audio, we
            include the WAV header. Note: as with all bytes fields,
            protobuffers use a pure binary representation, whereas JSON
            representations use base64.
    """
    audio_content: bytes = proto.Field(proto.BYTES, number=1)