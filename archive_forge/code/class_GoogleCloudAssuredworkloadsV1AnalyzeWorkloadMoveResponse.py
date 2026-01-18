from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAssuredworkloadsV1AnalyzeWorkloadMoveResponse(_messages.Message):
    """Response containing the analysis results for the hypothetical resource
  move.

  Fields:
    assetMoveAnalyses: List of analysis results for each asset in scope.
    nextPageToken: The next page token. Is empty if the last page is reached.
  """
    assetMoveAnalyses = _messages.MessageField('GoogleCloudAssuredworkloadsV1AssetMoveAnalysis', 1, repeated=True)
    nextPageToken = _messages.StringField(2)