from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAssuredworkloadsV1MoveAnalysisGroup(_messages.Message):
    """Represents a logical group of checks performed for an asset. If
  successful, the group contains the analysis result, otherwise it contains an
  error with the failure reason.

  Fields:
    analysisResult: Result of a successful analysis.
    displayName: Name of the analysis group.
    error: Error details for a failed analysis.
  """
    analysisResult = _messages.MessageField('GoogleCloudAssuredworkloadsV1MoveAnalysisResult', 1)
    displayName = _messages.StringField(2)
    error = _messages.MessageField('GoogleRpcStatus', 3)