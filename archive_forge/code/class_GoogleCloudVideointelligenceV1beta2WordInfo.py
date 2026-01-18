from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudVideointelligenceV1beta2WordInfo(_messages.Message):
    """Word-specific information for recognized words. Word information is only
  included in the response when certain request parameters are set, such as
  `enable_word_time_offsets`.

  Fields:
    confidence: Output only. The confidence estimate between 0.0 and 1.0. A
      higher number indicates an estimated greater likelihood that the
      recognized words are correct. This field is set only for the top
      alternative. This field is not guaranteed to be accurate and users
      should not rely on it to be always provided. The default of 0.0 is a
      sentinel value indicating `confidence` was not set.
    endTime: Time offset relative to the beginning of the audio, and
      corresponding to the end of the spoken word. This field is only set if
      `enable_word_time_offsets=true` and only in the top hypothesis. This is
      an experimental feature and the accuracy of the time offset can vary.
    speakerTag: Output only. A distinct integer value is assigned for every
      speaker within the audio. This field specifies which one of those
      speakers was detected to have spoken this word. Value ranges from 1 up
      to diarization_speaker_count, and is only set if speaker diarization is
      enabled.
    startTime: Time offset relative to the beginning of the audio, and
      corresponding to the start of the spoken word. This field is only set if
      `enable_word_time_offsets=true` and only in the top hypothesis. This is
      an experimental feature and the accuracy of the time offset can vary.
    word: The word corresponding to this set of information.
  """
    confidence = _messages.FloatField(1, variant=_messages.Variant.FLOAT)
    endTime = _messages.StringField(2)
    speakerTag = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    startTime = _messages.StringField(4)
    word = _messages.StringField(5)