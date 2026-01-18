from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudidentityGroupsUpdateSecuritySettingsRequest(_messages.Message):
    """A CloudidentityGroupsUpdateSecuritySettingsRequest object.

  Fields:
    name: Output only. The resource name of the security settings. Shall be of
      the form `groups/{group_id}/securitySettings`.
    securitySettings: A SecuritySettings resource to be passed as the request
      body.
    updateMask: Required. The fully-qualified names of fields to update. May
      only contain the following field: `member_restriction.query`.
  """
    name = _messages.StringField(1, required=True)
    securitySettings = _messages.MessageField('SecuritySettings', 2)
    updateMask = _messages.StringField(3)