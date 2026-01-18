from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OutputAudioEncodingValueValuesEnum(_messages.Enum):
    """Required. Audio encoding of the synthesized audio content.

    Values:
      OUTPUT_AUDIO_ENCODING_UNSPECIFIED: Not specified.
      OUTPUT_AUDIO_ENCODING_LINEAR_16: Uncompressed 16-bit signed little-
        endian samples (Linear PCM). Audio content returned as LINEAR16 also
        contains a WAV header.
      OUTPUT_AUDIO_ENCODING_MP3: MP3 audio at 32kbps.
      OUTPUT_AUDIO_ENCODING_MP3_64_KBPS: MP3 audio at 64kbps.
      OUTPUT_AUDIO_ENCODING_OGG_OPUS: Opus encoded audio wrapped in an ogg
        container. The result will be a file which can be played natively on
        Android, and in browsers (at least Chrome and Firefox). The quality of
        the encoding is considerably higher than MP3 while using approximately
        the same bitrate.
      OUTPUT_AUDIO_ENCODING_MULAW: 8-bit samples that compand 14-bit audio
        samples using G.711 PCMU/mu-law.
    """
    OUTPUT_AUDIO_ENCODING_UNSPECIFIED = 0
    OUTPUT_AUDIO_ENCODING_LINEAR_16 = 1
    OUTPUT_AUDIO_ENCODING_MP3 = 2
    OUTPUT_AUDIO_ENCODING_MP3_64_KBPS = 3
    OUTPUT_AUDIO_ENCODING_OGG_OPUS = 4
    OUTPUT_AUDIO_ENCODING_MULAW = 5