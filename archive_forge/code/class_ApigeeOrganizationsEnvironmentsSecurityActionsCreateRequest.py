from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsEnvironmentsSecurityActionsCreateRequest(_messages.Message):
    """A ApigeeOrganizationsEnvironmentsSecurityActionsCreateRequest object.

  Fields:
    googleCloudApigeeV1SecurityAction: A GoogleCloudApigeeV1SecurityAction
      resource to be passed as the request body.
    parent: Required. The organization and environment that this
      SecurityAction applies to. Format:
      organizations/{org}/environments/{env}
    securityActionId: Required. The ID to use for the SecurityAction, which
      will become the final component of the action's resource name. This
      value should be 0-61 characters, and valid format is
      (^[a-z]([a-z0-9-]{\\u200b0,61}[a-z0-9])?$).
  """
    googleCloudApigeeV1SecurityAction = _messages.MessageField('GoogleCloudApigeeV1SecurityAction', 1)
    parent = _messages.StringField(2, required=True)
    securityActionId = _messages.StringField(3)