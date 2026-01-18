from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IamFoldersLocationsPolicyBindingsPatchRequest(_messages.Message):
    """A IamFoldersLocationsPolicyBindingsPatchRequest object.

  Fields:
    googleIamV3betaPolicyBinding: A GoogleIamV3betaPolicyBinding resource to
      be passed as the request body.
    name: Identifier. The resource name of the policy binding. The binding
      parent is the closest CRM resource (i.e., Project, Folder or
      Organization) to the binding target. Format: `projects/{project_id}/loca
      tions/{location}/policyBindings/{policy_binding_id}` `projects/{project_
      number}/locations/{location}/policyBindings/{policy_binding_id}` `folder
      s/{folder_id}/locations/{location}/policyBindings/{policy_binding_id}` `
      organizations/{organization_id}/locations/{location}/policyBindings/{pol
      icy_binding_id}`
    updateMask: Optional. The list of fields to update
    validateOnly: Optional. If set, validate the request and preview the
      update, but do not actually post it.
  """
    googleIamV3betaPolicyBinding = _messages.MessageField('GoogleIamV3betaPolicyBinding', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)
    validateOnly = _messages.BooleanField(4)