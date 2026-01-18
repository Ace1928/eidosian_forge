from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAssetV1Access(_messages.Message):
    """An IAM role or permission under analysis.

  Fields:
    analysisState: The analysis state of this access.
    permission: The permission.
    role: The role.
  """
    analysisState = _messages.MessageField('IamPolicyAnalysisState', 1)
    permission = _messages.StringField(2)
    role = _messages.StringField(3)