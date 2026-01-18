from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsEnvironmentsUpdateSecurityActionsConfigRequest(_messages.Message):
    """A ApigeeOrganizationsEnvironmentsUpdateSecurityActionsConfigRequest
  object.

  Fields:
    googleCloudApigeeV1SecurityActionsConfig: A
      GoogleCloudApigeeV1SecurityActionsConfig resource to be passed as the
      request body.
    name: This is a singleton resource, the name will always be set by
      SecurityActions and any user input will be ignored. The name is always:
      `organizations/{org}/environments/{env}/security_actions_config`
    updateMask: The list of fields to update.
  """
    googleCloudApigeeV1SecurityActionsConfig = _messages.MessageField('GoogleCloudApigeeV1SecurityActionsConfig', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)