from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BwdifConfig(_messages.Message):
    """Bob Weaver Deinterlacing Filter Configuration.

  Fields:
    deinterlaceAllFrames: Deinterlace all frames rather than just the frames
      identified as interlaced. The default is `false`.
    mode: Specifies the deinterlacing mode to adopt. The default is
      `send_frame`. Supported values: - `send_frame`: Output one frame for
      each frame - `send_field`: Output one frame for each field
    parity: The picture field parity assumed for the input interlaced video.
      The default is `auto`. Supported values: - `tff`: Assume the top field
      is first - `bff`: Assume the bottom field is first - `auto`: Enable
      automatic detection of field parity
  """
    deinterlaceAllFrames = _messages.BooleanField(1)
    mode = _messages.StringField(2)
    parity = _messages.StringField(3)