from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsEnvironmentsSecurityActionsEnableRequest(_messages.Message):
    """A ApigeeOrganizationsEnvironmentsSecurityActionsEnableRequest object.

  Fields:
    googleCloudApigeeV1EnableSecurityActionRequest: A
      GoogleCloudApigeeV1EnableSecurityActionRequest resource to be passed as
      the request body.
    name: Required. The name of the SecurityAction to enable. Format:
      organizations/{org}/environments/{env}/securityActions/{security_action}
  """
    googleCloudApigeeV1EnableSecurityActionRequest = _messages.MessageField('GoogleCloudApigeeV1EnableSecurityActionRequest', 1)
    name = _messages.StringField(2, required=True)