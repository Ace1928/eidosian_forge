from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ImportGoogetArtifactsErrorInfo(_messages.Message):
    """Error information explaining why a package was not imported.

  Fields:
    error: The detailed error status.
    gcsSource: Google Cloud Storage location requested.
  """
    error = _messages.MessageField('Status', 1)
    gcsSource = _messages.MessageField('ImportGoogetArtifactsGcsSource', 2)