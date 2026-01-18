from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SpeechRecognitionResult(_messages.Message):
    """A speech recognition result corresponding to a portion of the audio.

  Fields:
    alternatives: May contain one or more recognition hypotheses. These
      alternatives are ordered in terms of accuracy, with the top (first)
      alternative being the most probable, as ranked by the recognizer.
    channelTag: For multi-channel audio, this is the channel number
      corresponding to the recognized result for the audio from that channel.
      For `audio_channel_count` = `N`, its output values can range from `1` to
      `N`.
    languageCode: Output only. The [BCP-47](https://www.rfc-
      editor.org/rfc/bcp/bcp47.txt) language tag of the language in this
      result. This language code was detected to have the most likelihood of
      being spoken in the audio.
    resultEndOffset: Time offset of the end of this result relative to the
      beginning of the audio.
  """
    alternatives = _messages.MessageField('SpeechRecognitionAlternative', 1, repeated=True)
    channelTag = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    languageCode = _messages.StringField(3)
    resultEndOffset = _messages.StringField(4)