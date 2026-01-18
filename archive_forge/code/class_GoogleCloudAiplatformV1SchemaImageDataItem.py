from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1SchemaImageDataItem(_messages.Message):
    """Payload of Image DataItem.

  Fields:
    gcsUri: Required. Google Cloud Storage URI points to the original image in
      user's bucket. The image is up to 30MB in size.
    mimeType: Output only. The mime type of the content of the image. Only the
      images in below listed mime types are supported. - image/jpeg -
      image/gif - image/png - image/webp - image/bmp - image/tiff -
      image/vnd.microsoft.icon
  """
    gcsUri = _messages.StringField(1)
    mimeType = _messages.StringField(2)