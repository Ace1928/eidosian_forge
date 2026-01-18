from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IamProjectsLocationsPolicyBindingsDeleteRequest(_messages.Message):
    """A IamProjectsLocationsPolicyBindingsDeleteRequest object.

  Fields:
    etag: Optional. The etag of the policy binding. If this is provided, it
      must match the server's etag.
    name: Required. The name of the policy binding to delete. Format: `project
      s/{project_id}/locations/{location}/policyBindings/{policy_binding_id}`
      `projects/{project_number}/locations/{location}/policyBindings/{policy_b
      inding_id}` `folders/{folder_id}/locations/{location}/policyBindings/{po
      licy_binding_id}` `organizations/{organization_id}/locations/{location}/
      policyBindings/{policy_binding_id}`
    validateOnly: Optional. If set, validate the request and preview the
      deletion, but do not actually post it.
  """
    etag = _messages.StringField(1)
    name = _messages.StringField(2, required=True)
    validateOnly = _messages.BooleanField(3)