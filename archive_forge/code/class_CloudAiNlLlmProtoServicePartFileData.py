from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudAiNlLlmProtoServicePartFileData(_messages.Message):
    """Represents file data.

  Fields:
    fileUri: Inline data.
    mimeType: The mime type corresponding to this input.
  """
    fileUri = _messages.StringField(1)
    mimeType = _messages.StringField(2)