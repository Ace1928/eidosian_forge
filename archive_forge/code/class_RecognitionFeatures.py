from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RecognitionFeatures(_messages.Message):
    """Available recognition features.

  Enums:
    MultiChannelModeValueValuesEnum: Mode for recognizing multi-channel audio.

  Fields:
    diarizationConfig: Configuration to enable speaker diarization and set
      additional parameters to make diarization better suited for your
      application. When this is enabled, we send all the words from the
      beginning of the audio for the top alternative in every consecutive
      STREAMING responses. This is done in order to improve our speaker tags
      as our models learn to identify the speakers in the conversation over
      time. For non-streaming requests, the diarization results will be
      provided only in the top alternative of the FINAL
      SpeechRecognitionResult.
    enableAutomaticPunctuation: If `true`, adds punctuation to recognition
      result hypotheses. This feature is only available in select languages.
      The default `false` value does not add punctuation to result hypotheses.
    enableSpokenEmojis: The spoken emoji behavior for the call. If `true`,
      adds spoken emoji formatting for the request. This will replace spoken
      emojis with the corresponding Unicode symbols in the final transcript.
      If `false`, spoken emojis are not replaced.
    enableSpokenPunctuation: The spoken punctuation behavior for the call. If
      `true`, replaces spoken punctuation with the corresponding symbols in
      the request. For example, "how are you question mark" becomes "how are
      you?". See https://cloud.google.com/speech-to-text/docs/spoken-
      punctuation for support. If `false`, spoken punctuation is not replaced.
    enableWordConfidence: If `true`, the top result includes a list of words
      and the confidence for those words. If `false`, no word-level confidence
      information is returned. The default is `false`.
    enableWordTimeOffsets: If `true`, the top result includes a list of words
      and the start and end time offsets (timestamps) for those words. If
      `false`, no word-level time offset information is returned. The default
      is `false`.
    maxAlternatives: Maximum number of recognition hypotheses to be returned.
      The server may return fewer than `max_alternatives`. Valid values are
      `0`-`30`. A value of `0` or `1` will return a maximum of one. If
      omitted, will return a maximum of one.
    multiChannelMode: Mode for recognizing multi-channel audio.
    profanityFilter: If set to `true`, the server will attempt to filter out
      profanities, replacing all but the initial character in each filtered
      word with asterisks, for instance, "f***". If set to `false` or omitted,
      profanities won't be filtered out.
  """

    class MultiChannelModeValueValuesEnum(_messages.Enum):
        """Mode for recognizing multi-channel audio.

    Values:
      MULTI_CHANNEL_MODE_UNSPECIFIED: Default value for the multi-channel
        mode. If the audio contains multiple channels, only the first channel
        will be transcribed; other channels will be ignored.
      SEPARATE_RECOGNITION_PER_CHANNEL: If selected, each channel in the
        provided audio is transcribed independently. This cannot be selected
        if the selected model is `latest_short`.
    """
        MULTI_CHANNEL_MODE_UNSPECIFIED = 0
        SEPARATE_RECOGNITION_PER_CHANNEL = 1
    diarizationConfig = _messages.MessageField('SpeakerDiarizationConfig', 1)
    enableAutomaticPunctuation = _messages.BooleanField(2)
    enableSpokenEmojis = _messages.BooleanField(3)
    enableSpokenPunctuation = _messages.BooleanField(4)
    enableWordConfidence = _messages.BooleanField(5)
    enableWordTimeOffsets = _messages.BooleanField(6)
    maxAlternatives = _messages.IntegerField(7, variant=_messages.Variant.INT32)
    multiChannelMode = _messages.EnumField('MultiChannelModeValueValuesEnum', 8)
    profanityFilter = _messages.BooleanField(9)