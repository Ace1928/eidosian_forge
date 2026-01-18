from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ExplicitDecodingConfig(_messages.Message):
    """Explicitly specified decoding parameters.

  Enums:
    EncodingValueValuesEnum: Required. Encoding of the audio data sent for
      recognition.

  Fields:
    audioChannelCount: Number of channels present in the audio data sent for
      recognition. Supported for the following encodings: * LINEAR16:
      Headerless 16-bit signed little-endian PCM samples. * MULAW: Headerless
      8-bit companded mulaw samples. * ALAW: Headerless 8-bit companded alaw
      samples. The maximum allowed value is 8.
    encoding: Required. Encoding of the audio data sent for recognition.
    sampleRateHertz: Sample rate in Hertz of the audio data sent for
      recognition. Valid values are: 8000-48000. 16000 is optimal. For best
      results, set the sampling rate of the audio source to 16000 Hz. If
      that's not possible, use the native sample rate of the audio source
      (instead of re-sampling). Supported for the following encodings: *
      LINEAR16: Headerless 16-bit signed little-endian PCM samples. * MULAW:
      Headerless 8-bit companded mulaw samples. * ALAW: Headerless 8-bit
      companded alaw samples.
  """

    class EncodingValueValuesEnum(_messages.Enum):
        """Required. Encoding of the audio data sent for recognition.

    Values:
      AUDIO_ENCODING_UNSPECIFIED: Default value. This value is unused.
      LINEAR16: Headerless 16-bit signed little-endian PCM samples.
      MULAW: Headerless 8-bit companded mulaw samples.
      ALAW: Headerless 8-bit companded alaw samples.
    """
        AUDIO_ENCODING_UNSPECIFIED = 0
        LINEAR16 = 1
        MULAW = 2
        ALAW = 3
    audioChannelCount = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    encoding = _messages.EnumField('EncodingValueValuesEnum', 2)
    sampleRateHertz = _messages.IntegerField(3, variant=_messages.Variant.INT32)