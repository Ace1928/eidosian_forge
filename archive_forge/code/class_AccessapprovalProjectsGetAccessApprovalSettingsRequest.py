from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class AccessapprovalProjectsGetAccessApprovalSettingsRequest(_messages.Message):
    """A AccessapprovalProjectsGetAccessApprovalSettingsRequest object.

  Fields:
    name: The name of the AccessApprovalSettings to retrieve. Format:
      "{projects|folders|organizations}/{id}/accessApprovalSettings"
  """
    name = _messages.StringField(1, required=True)