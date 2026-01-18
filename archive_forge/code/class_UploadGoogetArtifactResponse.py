from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UploadGoogetArtifactResponse(_messages.Message):
    """The response of the completed artifact upload operation. This response
  is contained in the Operation and available to users.

  Fields:
    googetArtifacts: The GooGet artifacts updated.
  """
    googetArtifacts = _messages.MessageField('GoogetArtifact', 1, repeated=True)