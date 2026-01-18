from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudAiNlLlmProtoServicePartBlob(_messages.Message):
    """Represents arbitrary blob data input.

  Fields:
    data: Inline data.
    mimeType: The mime type corresponding to this input.
    originalFileData: Original file data where the blob comes from.
  """
    data = _messages.BytesField(1)
    mimeType = _messages.StringField(2)
    originalFileData = _messages.MessageField('CloudAiNlLlmProtoServicePartFileData', 3)