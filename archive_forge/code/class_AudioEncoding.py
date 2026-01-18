from __future__ import annotations
from typing import MutableMapping, MutableSequence
from google.protobuf import duration_pb2  # type: ignore
from google.protobuf import timestamp_pb2  # type: ignore
from google.protobuf import wrappers_pb2  # type: ignore
from google.rpc import status_pb2  # type: ignore
import proto  # type: ignore
from google.cloud.speech_v1p1beta1.types import resource
class AudioEncoding(proto.Enum):
    """The encoding of the audio data sent in the request.

        All encodings support only 1 channel (mono) audio, unless the
        ``audio_channel_count`` and
        ``enable_separate_recognition_per_channel`` fields are set.

        For best results, the audio source should be captured and
        transmitted using a lossless encoding (``FLAC`` or ``LINEAR16``).
        The accuracy of the speech recognition can be reduced if lossy
        codecs are used to capture or transmit audio, particularly if
        background noise is present. Lossy codecs include ``MULAW``,
        ``AMR``, ``AMR_WB``, ``OGG_OPUS``, ``SPEEX_WITH_HEADER_BYTE``,
        ``MP3``, and ``WEBM_OPUS``.

        The ``FLAC`` and ``WAV`` audio file formats include a header that
        describes the included audio content. You can request recognition
        for ``WAV`` files that contain either ``LINEAR16`` or ``MULAW``
        encoded audio. If you send ``FLAC`` or ``WAV`` audio file format in
        your request, you do not need to specify an ``AudioEncoding``; the
        audio encoding format is determined from the file header. If you
        specify an ``AudioEncoding`` when you send send ``FLAC`` or ``WAV``
        audio, the encoding configuration must match the encoding described
        in the audio header; otherwise the request returns an
        [google.rpc.Code.INVALID_ARGUMENT][google.rpc.Code.INVALID_ARGUMENT]
        error code.

        Values:
            ENCODING_UNSPECIFIED (0):
                Not specified.
            LINEAR16 (1):
                Uncompressed 16-bit signed little-endian
                samples (Linear PCM).
            FLAC (2):
                ``FLAC`` (Free Lossless Audio Codec) is the recommended
                encoding because it is lossless--therefore recognition is
                not compromised--and requires only about half the bandwidth
                of ``LINEAR16``. ``FLAC`` stream encoding supports 16-bit
                and 24-bit samples, however, not all fields in
                ``STREAMINFO`` are supported.
            MULAW (3):
                8-bit samples that compand 14-bit audio
                samples using G.711 PCMU/mu-law.
            AMR (4):
                Adaptive Multi-Rate Narrowband codec. ``sample_rate_hertz``
                must be 8000.
            AMR_WB (5):
                Adaptive Multi-Rate Wideband codec. ``sample_rate_hertz``
                must be 16000.
            OGG_OPUS (6):
                Opus encoded audio frames in Ogg container
                (`OggOpus <https://wiki.xiph.org/OggOpus>`__).
                ``sample_rate_hertz`` must be one of 8000, 12000, 16000,
                24000, or 48000.
            SPEEX_WITH_HEADER_BYTE (7):
                Although the use of lossy encodings is not recommended, if a
                very low bitrate encoding is required, ``OGG_OPUS`` is
                highly preferred over Speex encoding. The
                `Speex <https://speex.org/>`__ encoding supported by Cloud
                Speech API has a header byte in each block, as in MIME type
                ``audio/x-speex-with-header-byte``. It is a variant of the
                RTP Speex encoding defined in `RFC
                5574 <https://tools.ietf.org/html/rfc5574>`__. The stream is
                a sequence of blocks, one block per RTP packet. Each block
                starts with a byte containing the length of the block, in
                bytes, followed by one or more frames of Speex data, padded
                to an integral number of bytes (octets) as specified in RFC
                5574. In other words, each RTP header is replaced with a
                single byte containing the block length. Only Speex wideband
                is supported. ``sample_rate_hertz`` must be 16000.
            MP3 (8):
                MP3 audio. MP3 encoding is a Beta feature and only available
                in v1p1beta1. Support all standard MP3 bitrates (which range
                from 32-320 kbps). When using this encoding,
                ``sample_rate_hertz`` has to match the sample rate of the
                file being used.
            WEBM_OPUS (9):
                Opus encoded audio frames in WebM container
                (`OggOpus <https://wiki.xiph.org/OggOpus>`__).
                ``sample_rate_hertz`` must be one of 8000, 12000, 16000,
                24000, or 48000.
        """
    ENCODING_UNSPECIFIED = 0
    LINEAR16 = 1
    FLAC = 2
    MULAW = 3
    AMR = 4
    AMR_WB = 5
    OGG_OPUS = 6
    SPEEX_WITH_HEADER_BYTE = 7
    MP3 = 8
    WEBM_OPUS = 9