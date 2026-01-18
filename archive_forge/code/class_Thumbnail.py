from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Thumbnail(_messages.Message):
    """A single thumbnail, with its size and format.

  Fields:
    contentType: The thumbnail's content type, i.e. "image/png". Always set.
    data: The thumbnail file itself. That is, the bytes here are precisely the
      bytes that make up the thumbnail file; they can be served as an image
      as-is (with the appropriate content type.) Always set.
    heightPx: The height of the thumbnail, in pixels. Always set.
    widthPx: The width of the thumbnail, in pixels. Always set.
  """
    contentType = _messages.StringField(1)
    data = _messages.BytesField(2)
    heightPx = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    widthPx = _messages.IntegerField(4, variant=_messages.Variant.INT32)