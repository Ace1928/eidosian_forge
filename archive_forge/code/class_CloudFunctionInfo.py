from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudFunctionInfo(_messages.Message):
    """For display only. Metadata associated with a Cloud Function.

  Fields:
    displayName: Name of a Cloud Function.
    location: Location in which the Cloud Function is deployed.
    uri: URI of a Cloud Function.
    versionId: Latest successfully deployed version id of the Cloud Function.
  """
    displayName = _messages.StringField(1)
    location = _messages.StringField(2)
    uri = _messages.StringField(3)
    versionId = _messages.IntegerField(4)