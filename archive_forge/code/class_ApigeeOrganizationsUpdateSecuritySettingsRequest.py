from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsUpdateSecuritySettingsRequest(_messages.Message):
    """A ApigeeOrganizationsUpdateSecuritySettingsRequest object.

  Fields:
    googleCloudApigeeV1SecuritySettings: A GoogleCloudApigeeV1SecuritySettings
      resource to be passed as the request body.
    name: Identifier. Full resource name is always
      `organizations/{org}/securitySettings`.
    updateMask: Optional. The list of fields to update. Allowed fields are: -
      ml_retraining_feedback_enabled
  """
    googleCloudApigeeV1SecuritySettings = _messages.MessageField('GoogleCloudApigeeV1SecuritySettings', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)