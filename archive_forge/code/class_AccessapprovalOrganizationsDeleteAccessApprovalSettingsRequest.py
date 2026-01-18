from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class AccessapprovalOrganizationsDeleteAccessApprovalSettingsRequest(_messages.Message):
    """A AccessapprovalOrganizationsDeleteAccessApprovalSettingsRequest object.

  Fields:
    name: Name of the AccessApprovalSettings to delete.
  """
    name = _messages.StringField(1, required=True)