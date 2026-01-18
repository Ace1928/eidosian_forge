from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AnalyzePackagesMetadata(_messages.Message):
    """AnalyzePackagesMetadata contains metadata for an active scan of a
  container image.

  Fields:
    createTime: When the scan was created.
    resourceUri: The resource URI of the container image being scanned.
  """
    createTime = _messages.StringField(1)
    resourceUri = _messages.StringField(2)