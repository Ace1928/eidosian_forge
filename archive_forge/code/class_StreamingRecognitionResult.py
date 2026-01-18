from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StreamingRecognitionResult(_messages.Message):
    """A streaming speech recognition result corresponding to a portion of the
  audio that is currently being processed.

  Fields:
    alternatives: May contain one or more recognition hypotheses. These
      alternatives are ordered in terms of accuracy, with the top (first)
      alternative being the most probable, as ranked by the recognizer.
    channelTag: For multi-channel audio, this is the channel number
      corresponding to the recognized result for the audio from that channel.
      For `audio_channel_count` = `N`, its output values can range from `1` to
      `N`.
    isFinal: If `false`, this StreamingRecognitionResult represents an interim
      result that may change. If `true`, this is the final time the speech
      service will return this particular StreamingRecognitionResult, the
      recognizer will not return any further hypotheses for this portion of
      the transcript and corresponding audio.
    languageCode: Output only. The [BCP-47](https://www.rfc-
      editor.org/rfc/bcp/bcp47.txt) language tag of the language in this
      result. This language code was detected to have the most likelihood of
      being spoken in the audio.
    resultEndOffset: Time offset of the end of this result relative to the
      beginning of the audio.
    stability: An estimate of the likelihood that the recognizer will not
      change its guess about this interim result. Values range from 0.0
      (completely unstable) to 1.0 (completely stable). This field is only
      provided for interim results (is_final=`false`). The default of 0.0 is a
      sentinel value indicating `stability` was not set.
  """
    alternatives = _messages.MessageField('SpeechRecognitionAlternative', 1, repeated=True)
    channelTag = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    isFinal = _messages.BooleanField(3)
    languageCode = _messages.StringField(4)
    resultEndOffset = _messages.StringField(5)
    stability = _messages.FloatField(6, variant=_messages.Variant.FLOAT)