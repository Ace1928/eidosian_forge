from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IamPoliciesCreatePolicyRequest(_messages.Message):
    """A IamPoliciesCreatePolicyRequest object.

  Fields:
    googleIamV2Policy: A GoogleIamV2Policy resource to be passed as the
      request body.
    parent: Required. The resource that the policy is attached to, along with
      the kind of policy to create. Format:
      `policies/{attachment_point}/denypolicies` The attachment point is
      identified by its URL-encoded full resource name, which means that the
      forward-slash character, `/`, must be written as `%2F`. For example,
      `policies/cloudresourcemanager.googleapis.com%2Fprojects%2Fmy-
      project/denypolicies`. For organizations and folders, use the numeric ID
      in the full resource name. For projects, you can use the alphanumeric or
      the numeric ID.
    policyId: The ID to use for this policy, which will become the final
      component of the policy's resource name. The ID must contain 3 to 63
      characters. It can contain lowercase letters and numbers, as well as
      dashes (`-`) and periods (`.`). The first character must be a lowercase
      letter.
  """
    googleIamV2Policy = _messages.MessageField('GoogleIamV2Policy', 1)
    parent = _messages.StringField(2, required=True)
    policyId = _messages.StringField(3)