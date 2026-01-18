from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
class SsmlVoiceGender(proto.Enum):
    """Gender of the voice as described in `SSML voice
    element <https://www.w3.org/TR/speech-synthesis11/#edef_voice>`__.

    Values:
        SSML_VOICE_GENDER_UNSPECIFIED (0):
            An unspecified gender.
            In VoiceSelectionParams, this means that the
            client doesn't care which gender the selected
            voice will have. In the Voice field of
            ListVoicesResponse, this may mean that the voice
            doesn't fit any of the other categories in this
            enum, or that the gender of the voice isn't
            known.
        MALE (1):
            A male voice.
        FEMALE (2):
            A female voice.
        NEUTRAL (3):
            A gender-neutral voice. This voice is not yet
            supported.
    """
    SSML_VOICE_GENDER_UNSPECIFIED = 0
    MALE = 1
    FEMALE = 2
    NEUTRAL = 3