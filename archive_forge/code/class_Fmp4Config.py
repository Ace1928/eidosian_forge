from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Fmp4Config(_messages.Message):
    """`fmp4` container configuration.

  Fields:
    codecTag: Optional. Specify the codec tag string that will be used in the
      media bitstream. When not specified, the codec appropriate value is
      used. Supported H265 codec tags: - `hvc1` (default) - `hev1`
  """
    codecTag = _messages.StringField(1)