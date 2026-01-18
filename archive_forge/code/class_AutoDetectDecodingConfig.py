from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AutoDetectDecodingConfig(_messages.Message):
    """Automatically detected decoding parameters. Supported for the following
  encodings: * WAV_LINEAR16: 16-bit signed little-endian PCM samples in a WAV
  container. * WAV_MULAW: 8-bit companded mulaw samples in a WAV container. *
  WAV_ALAW: 8-bit companded alaw samples in a WAV container. * RFC4867_5_AMR:
  AMR frames with an rfc4867.5 header. * RFC4867_5_AMRWB: AMR-WB frames with
  an rfc4867.5 header. * FLAC: FLAC frames in the "native FLAC" container
  format. * MP3: MPEG audio frames with optional (ignored) ID3 metadata. *
  OGG_OPUS: Opus audio frames in an Ogg container. * WEBM_OPUS: Opus audio
  frames in a WebM container. * M4A: M4A audio format.
  """