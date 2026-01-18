from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class AccessapprovalOrganizationsUpdateAccessApprovalSettingsRequest(_messages.Message):
    """A AccessapprovalOrganizationsUpdateAccessApprovalSettingsRequest object.

  Fields:
    accessApprovalSettings: A AccessApprovalSettings resource to be passed as
      the request body.
    name: The resource name of the settings. Format is one of: *
      "projects/{project}/accessApprovalSettings" *
      "folders/{folder}/accessApprovalSettings" *
      "organizations/{organization}/accessApprovalSettings"
    updateMask: The update mask applies to the settings. Only the top level
      fields of AccessApprovalSettings (notification_emails &
      enrolled_services) are supported. For each field, if it is included, the
      currently stored value will be entirely overwritten with the value of
      the field passed in this request. For the `FieldMask` definition, see
      https://developers.google.com/protocol-
      buffers/docs/reference/google.protobuf#fieldmask If this field is left
      unset, only the notification_emails field will be updated.
  """
    accessApprovalSettings = _messages.MessageField('AccessApprovalSettings', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)