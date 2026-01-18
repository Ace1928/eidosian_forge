from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsEnvironmentsFlowhooksAttachSharedFlowToFlowHookRequest(_messages.Message):
    """A
  ApigeeOrganizationsEnvironmentsFlowhooksAttachSharedFlowToFlowHookRequest
  object.

  Fields:
    googleCloudApigeeV1FlowHook: A GoogleCloudApigeeV1FlowHook resource to be
      passed as the request body.
    name: Required. Name of the flow hook to which the shared flow should be
      attached in the following format:
      `organizations/{org}/environments/{env}/flowhooks/{flowhook}`
  """
    googleCloudApigeeV1FlowHook = _messages.MessageField('GoogleCloudApigeeV1FlowHook', 1)
    name = _messages.StringField(2, required=True)