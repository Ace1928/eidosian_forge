from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class GoogleCloudAssetV1p4alpha1Resource(_messages.Message):
    """A GCP resource that appears in an access control list.

  Fields:
    analysisState: The analysis state of this resource node.
    fullResourceName: The [full resource name](https://aip.dev/122#full-
      resource-names).
  """
    analysisState = _messages.MessageField('GoogleCloudAssetV1p4alpha1AnalysisState', 1)
    fullResourceName = _messages.StringField(2)