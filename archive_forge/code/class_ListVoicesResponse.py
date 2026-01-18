from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
class ListVoicesResponse(proto.Message):
    """The message returned to the client by the ``ListVoices`` method.

    Attributes:
        voices (MutableSequence[google.cloud.texttospeech_v1.types.Voice]):
            The list of voices.
    """
    voices: MutableSequence['Voice'] = proto.RepeatedField(proto.MESSAGE, number=1, message='Voice')