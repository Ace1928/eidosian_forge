from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ImportGoogetArtifactsResponse(_messages.Message):
    """The response message from importing artifacts.

  Fields:
    errors: Detailed error info for packages that were not imported.
    googetArtifacts: The GooGet artifacts updated.
  """
    errors = _messages.MessageField('ImportGoogetArtifactsErrorInfo', 1, repeated=True)
    googetArtifacts = _messages.MessageField('GoogetArtifact', 2, repeated=True)