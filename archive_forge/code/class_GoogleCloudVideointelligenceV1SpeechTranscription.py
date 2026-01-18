from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudVideointelligenceV1SpeechTranscription(_messages.Message):
    """A speech recognition result corresponding to a portion of the audio.

  Fields:
    alternatives: May contain one or more recognition hypotheses (up to the
      maximum specified in `max_alternatives`). These alternatives are ordered
      in terms of accuracy, with the top (first) alternative being the most
      probable, as ranked by the recognizer.
    languageCode: Output only. The [BCP-47](https://www.rfc-
      editor.org/rfc/bcp/bcp47.txt) language tag of the language in this
      result. This language code was detected to have the most likelihood of
      being spoken in the audio.
  """
    alternatives = _messages.MessageField('GoogleCloudVideointelligenceV1SpeechRecognitionAlternative', 1, repeated=True)
    languageCode = _messages.StringField(2)