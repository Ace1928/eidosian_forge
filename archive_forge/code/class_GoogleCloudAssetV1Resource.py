from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAssetV1Resource(_messages.Message):
    """A Google Cloud resource under analysis.

  Fields:
    analysisState: The analysis state of this resource.
    fullResourceName: The [full resource name](https://cloud.google.com/asset-
      inventory/docs/resource-name-format)
  """
    analysisState = _messages.MessageField('IamPolicyAnalysisState', 1)
    fullResourceName = _messages.StringField(2)