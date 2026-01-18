from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class GoogleCloudAssetV1p4alpha1Access(_messages.Message):
    """A role or permission that appears in an access control list.

  Fields:
    analysisState: The analysis state of this access node.
    permission: The permission.
    role: The role.
  """
    analysisState = _messages.MessageField('GoogleCloudAssetV1p4alpha1AnalysisState', 1)
    permission = _messages.StringField(2)
    role = _messages.StringField(3)