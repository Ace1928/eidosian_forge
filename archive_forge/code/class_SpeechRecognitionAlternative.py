from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SpeechRecognitionAlternative(_messages.Message):
    """Alternative hypotheses (a.k.a. n-best list).

  Fields:
    confidence: The confidence estimate between 0.0 and 1.0. A higher number
      indicates an estimated greater likelihood that the recognized words are
      correct. This field is set only for the top alternative of a non-
      streaming result or, of a streaming result where is_final is set to
      `true`. This field is not guaranteed to be accurate and users should not
      rely on it to be always provided. The default of 0.0 is a sentinel value
      indicating `confidence` was not set.
    transcript: Transcript text representing the words that the user spoke.
    words: A list of word-specific information for each recognized word. When
      the SpeakerDiarizationConfig is set, you will see all the words from the
      beginning of the audio.
  """
    confidence = _messages.FloatField(1, variant=_messages.Variant.FLOAT)
    transcript = _messages.StringField(2)
    words = _messages.MessageField('WordInfo', 3, repeated=True)