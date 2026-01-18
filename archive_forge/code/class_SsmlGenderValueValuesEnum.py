from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SsmlGenderValueValuesEnum(_messages.Enum):
    """Optional. The preferred gender of the voice. If not set, the service
    will choose a voice based on the other parameters such as language_code
    and name. Note that this is only a preference, not requirement. If a voice
    of the appropriate gender is not available, the synthesizer should
    substitute a voice with a different gender rather than failing the
    request.

    Values:
      SSML_VOICE_GENDER_UNSPECIFIED: An unspecified gender, which means that
        the client doesn't care which gender the selected voice will have.
      SSML_VOICE_GENDER_MALE: A male voice.
      SSML_VOICE_GENDER_FEMALE: A female voice.
      SSML_VOICE_GENDER_NEUTRAL: A gender-neutral voice.
    """
    SSML_VOICE_GENDER_UNSPECIFIED = 0
    SSML_VOICE_GENDER_MALE = 1
    SSML_VOICE_GENDER_FEMALE = 2
    SSML_VOICE_GENDER_NEUTRAL = 3