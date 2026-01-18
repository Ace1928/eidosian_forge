from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class OrgpolicyFoldersPoliciesPatchRequest(_messages.Message):
    """A OrgpolicyFoldersPoliciesPatchRequest object.

  Fields:
    googleCloudOrgpolicyV2Policy: A GoogleCloudOrgpolicyV2Policy resource to
      be passed as the request body.
    name: Immutable. The resource name of the policy. Must be one of the
      following forms, where `constraint_name` is the name of the constraint
      which this policy configures: *
      `projects/{project_number}/policies/{constraint_name}` *
      `folders/{folder_id}/policies/{constraint_name}` *
      `organizations/{organization_id}/policies/{constraint_name}` For
      example, `projects/123/policies/compute.disableSerialPortAccess`. Note:
      `projects/{project_id}/policies/{constraint_name}` is also an acceptable
      name for API requests, but responses will return the name using the
      equivalent project number.
    updateMask: Field mask used to specify the fields to be overwritten in the
      policy by the set. The fields specified in the update_mask are relative
      to the policy, not the full request.
  """
    googleCloudOrgpolicyV2Policy = _messages.MessageField('GoogleCloudOrgpolicyV2Policy', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)