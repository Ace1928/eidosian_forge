from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDocumentaiV1DocumentPageImage(_messages.Message):
    """Rendered image contents for this page.

  Fields:
    content: Raw byte content of the image.
    height: Height of the image in pixels.
    mimeType: Encoding [media type (MIME
      type)](https://www.iana.org/assignments/media-types/media-types.xhtml)
      for the image.
    width: Width of the image in pixels.
  """
    content = _messages.BytesField(1)
    height = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    mimeType = _messages.StringField(3)
    width = _messages.IntegerField(4, variant=_messages.Variant.INT32)