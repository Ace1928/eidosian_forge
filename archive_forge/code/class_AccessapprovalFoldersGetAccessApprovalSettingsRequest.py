from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class AccessapprovalFoldersGetAccessApprovalSettingsRequest(_messages.Message):
    """A AccessapprovalFoldersGetAccessApprovalSettingsRequest object.

  Fields:
    name: The name of the AccessApprovalSettings to retrieve. Format:
      "{projects|folders|organizations}/{id}/accessApprovalSettings"
  """
    name = _messages.StringField(1, required=True)