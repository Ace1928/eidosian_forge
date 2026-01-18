from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class H265CodecSettings(_messages.Message):
    """H265 codec settings.

  Enums:
    FrameRateConversionStrategyValueValuesEnum: Optional. Frame rate
      conversion strategy for desired frame rate. The default is `DOWNSAMPLE`.

  Fields:
    allowOpenGop: Specifies whether an open Group of Pictures (GOP) structure
      should be allowed or not. The default is `false`.
    aqStrength: Specify the intensity of the adaptive quantizer (AQ). Must be
      between 0 and 1, where 0 disables the quantizer and 1 maximizes the
      quantizer. A higher value equals a lower bitrate but smoother image. The
      default is 0.
    bFrameCount: The number of consecutive B-frames. Must be greater than or
      equal to zero. Must be less than H265CodecSettings.gop_frame_count if
      set. The default is 0.
    bPyramid: Allow B-pyramid for reference frame selection. This may not be
      supported on all decoders. The default is `false`.
    bitrateBps: Required. The video bitrate in bits per second. The minimum
      value is 1,000. The maximum value is 800,000,000.
    crfLevel: Target CRF level. Must be between 10 and 36, where 10 is the
      highest quality and 36 is the most efficient compression. The default is
      21.
    enableTwoPass: Use two-pass encoding strategy to achieve better video
      quality. H265CodecSettings.rate_control_mode must be `vbr`. The default
      is `false`.
    frameRate: Required. The target video frame rate in frames per second
      (FPS). Must be less than or equal to 120.
    frameRateConversionStrategy: Optional. Frame rate conversion strategy for
      desired frame rate. The default is `DOWNSAMPLE`.
    gopDuration: Select the GOP size based on the specified duration. The
      default is `3s`. Note that `gopDuration` must be less than or equal to
      [`segmentDuration`](#SegmentSettings), and
      [`segmentDuration`](#SegmentSettings) must be divisible by
      `gopDuration`.
    gopFrameCount: Select the GOP size based on the specified frame count.
      Must be greater than zero.
    hdr10: Optional. HDR10 color format setting for H265.
    heightPixels: The height of the video in pixels. Must be an even integer.
      When not specified, the height is adjusted to match the specified width
      and input aspect ratio. If both are omitted, the input height is used.
      For portrait videos that contain horizontal ASR and rotation metadata,
      provide the height, in pixels, per the horizontal ASR. The API
      calculates the width per the horizontal ASR. The API detects any
      rotation metadata and swaps the requested height and width for the
      output.
    hlg: Optional. HLG color format setting for H265.
    pixelFormat: Pixel format to use. The default is `yuv420p`. Supported
      pixel formats: - `yuv420p` pixel format - `yuv422p` pixel format -
      `yuv444p` pixel format - `yuv420p10` 10-bit HDR pixel format -
      `yuv422p10` 10-bit HDR pixel format - `yuv444p10` 10-bit HDR pixel
      format - `yuv420p12` 12-bit HDR pixel format - `yuv422p12` 12-bit HDR
      pixel format - `yuv444p12` 12-bit HDR pixel format
    preset: Enforces the specified codec preset. The default is `veryfast`.
      The available options are [FFmpeg-
      compatible](https://trac.ffmpeg.org/wiki/Encode/H.265). Note that
      certain values for this field may cause the transcoder to override other
      fields you set in the `H265CodecSettings` message.
    profile: Enforces the specified codec profile. The following profiles are
      supported: * 8-bit profiles * `main` (default) * `main-intra` *
      `mainstillpicture` * 10-bit profiles * `main10` (default) *
      `main10-intra` * `main422-10` * `main422-10-intra` * `main444-10` *
      `main444-10-intra` * 12-bit profiles * `main12` (default) *
      `main12-intra` * `main422-12` * `main422-12-intra` * `main444-12` *
      `main444-12-intra` The available options are [FFmpeg-
      compatible](https://x265.readthedocs.io/). Note that certain values for
      this field may cause the transcoder to override other fields you set in
      the `H265CodecSettings` message.
    rateControlMode: Specify the mode. The default is `vbr`. Supported rate
      control modes: - `vbr` - variable bitrate - `crf` - constant rate factor
    sdr: Optional. SDR color format setting for H265.
    tune: Enforces the specified codec tune. The available options are
      [FFmpeg-compatible](https://trac.ffmpeg.org/wiki/Encode/H.265). Note
      that certain values for this field may cause the transcoder to override
      other fields you set in the `H265CodecSettings` message.
    vbvFullnessBits: Initial fullness of the Video Buffering Verifier (VBV)
      buffer in bits. Must be greater than zero. The default is equal to 90%
      of H265CodecSettings.vbv_size_bits.
    vbvSizeBits: Size of the Video Buffering Verifier (VBV) buffer in bits.
      Must be greater than zero. The default is equal to
      `VideoStream.bitrate_bps`.
    widthPixels: The width of the video in pixels. Must be an even integer.
      When not specified, the width is adjusted to match the specified height
      and input aspect ratio. If both are omitted, the input width is used.
      For portrait videos that contain horizontal ASR and rotation metadata,
      provide the width, in pixels, per the horizontal ASR. The API calculates
      the height per the horizontal ASR. The API detects any rotation metadata
      and swaps the requested height and width for the output.
  """

    class FrameRateConversionStrategyValueValuesEnum(_messages.Enum):
        """Optional. Frame rate conversion strategy for desired frame rate. The
    default is `DOWNSAMPLE`.

    Values:
      FRAME_RATE_CONVERSION_STRATEGY_UNSPECIFIED: Unspecified frame rate
        conversion strategy.
      DOWNSAMPLE: Selectively retain frames to reduce the output frame rate.
        Every _n_ th frame is kept, where `n = ceil(input frame rate / target
        frame rate)`. When _n_ = 1 (that is, the target frame rate is greater
        than the input frame rate), the output frame rate matches the input
        frame rate. When _n_ > 1, frames are dropped and the output frame rate
        is equal to `(input frame rate / n)`. For more information, see
        [Calculate frame
        rate](https://cloud.google.com/transcoder/docs/concepts/frame-rate).
      DROP_DUPLICATE: Drop or duplicate frames to match the specified frame
        rate.
    """
        FRAME_RATE_CONVERSION_STRATEGY_UNSPECIFIED = 0
        DOWNSAMPLE = 1
        DROP_DUPLICATE = 2
    allowOpenGop = _messages.BooleanField(1)
    aqStrength = _messages.FloatField(2)
    bFrameCount = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    bPyramid = _messages.BooleanField(4)
    bitrateBps = _messages.IntegerField(5, variant=_messages.Variant.INT32)
    crfLevel = _messages.IntegerField(6, variant=_messages.Variant.INT32)
    enableTwoPass = _messages.BooleanField(7)
    frameRate = _messages.FloatField(8)
    frameRateConversionStrategy = _messages.EnumField('FrameRateConversionStrategyValueValuesEnum', 9)
    gopDuration = _messages.StringField(10)
    gopFrameCount = _messages.IntegerField(11, variant=_messages.Variant.INT32)
    hdr10 = _messages.MessageField('H265ColorFormatHDR10', 12)
    heightPixels = _messages.IntegerField(13, variant=_messages.Variant.INT32)
    hlg = _messages.MessageField('H265ColorFormatHLG', 14)
    pixelFormat = _messages.StringField(15)
    preset = _messages.StringField(16)
    profile = _messages.StringField(17)
    rateControlMode = _messages.StringField(18)
    sdr = _messages.MessageField('H265ColorFormatSDR', 19)
    tune = _messages.StringField(20)
    vbvFullnessBits = _messages.IntegerField(21, variant=_messages.Variant.INT32)
    vbvSizeBits = _messages.IntegerField(22, variant=_messages.Variant.INT32)
    widthPixels = _messages.IntegerField(23, variant=_messages.Variant.INT32)