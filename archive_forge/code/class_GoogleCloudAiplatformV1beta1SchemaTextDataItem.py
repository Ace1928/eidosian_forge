from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1SchemaTextDataItem(_messages.Message):
    """Payload of Text DataItem.

  Fields:
    gcsUri: Output only. Google Cloud Storage URI points to the original text
      in user's bucket. The text file is up to 10MB in size.
  """
    gcsUri = _messages.StringField(1)