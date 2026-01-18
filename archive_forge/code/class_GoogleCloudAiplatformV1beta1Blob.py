from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1Blob(_messages.Message):
    """Content blob. It's preferred to send as text directly rather than raw
  bytes.

  Fields:
    data: Required. Raw bytes.
    mimeType: Required. The IANA standard MIME type of the source data.
  """
    data = _messages.BytesField(1)
    mimeType = _messages.StringField(2)