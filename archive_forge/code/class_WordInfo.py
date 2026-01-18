from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class WordInfo(_messages.Message):
    """Word-specific information for recognized words.

  Fields:
    confidence: The confidence estimate between 0.0 and 1.0. A higher number
      indicates an estimated greater likelihood that the recognized words are
      correct. This field is set only for the top alternative of a non-
      streaming result or, of a streaming result where is_final is set to
      `true`. This field is not guaranteed to be accurate and users should not
      rely on it to be always provided. The default of 0.0 is a sentinel value
      indicating `confidence` was not set.
    endOffset: Time offset relative to the beginning of the audio, and
      corresponding to the end of the spoken word. This field is only set if
      enable_word_time_offsets is `true` and only in the top hypothesis. This
      is an experimental feature and the accuracy of the time offset can vary.
    speakerLabel: A distinct label is assigned for every speaker within the
      audio. This field specifies which one of those speakers was detected to
      have spoken this word. `speaker_label` is set if
      SpeakerDiarizationConfig is given and only in the top alternative.
    startOffset: Time offset relative to the beginning of the audio, and
      corresponding to the start of the spoken word. This field is only set if
      enable_word_time_offsets is `true` and only in the top hypothesis. This
      is an experimental feature and the accuracy of the time offset can vary.
    word: The word corresponding to this set of information.
  """
    confidence = _messages.FloatField(1, variant=_messages.Variant.FLOAT)
    endOffset = _messages.StringField(2)
    speakerLabel = _messages.StringField(3)
    startOffset = _messages.StringField(4)
    word = _messages.StringField(5)