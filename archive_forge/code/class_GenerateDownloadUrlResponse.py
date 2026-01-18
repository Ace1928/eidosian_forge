from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GenerateDownloadUrlResponse(_messages.Message):
    """Response of `GenerateDownloadUrl` method.

  Fields:
    downloadUrl: The generated Google Cloud Storage signed URL that should be
      used for function source code download.
  """
    downloadUrl = _messages.StringField(1)