from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PreviewResult(_messages.Message):
    """Contains a signed Cloud Storage URLs.

  Fields:
    binarySignedUri: Output only. Plan binary signed URL
    jsonSignedUri: Output only. Plan JSON signed URL
  """
    binarySignedUri = _messages.StringField(1)
    jsonSignedUri = _messages.StringField(2)