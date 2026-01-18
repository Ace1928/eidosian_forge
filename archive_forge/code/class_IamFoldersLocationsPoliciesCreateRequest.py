from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IamFoldersLocationsPoliciesCreateRequest(_messages.Message):
    """A IamFoldersLocationsPoliciesCreateRequest object.

  Fields:
    googleIamV3betaV3Policy: A GoogleIamV3betaV3Policy resource to be passed
      as the request body.
    parent: Required. The parent of the new Policy. The parent needs to follow
      formats below. `projects/{project_id}/locations/{location}`
      `projects/{project_number}/locations/{location}`
      `folders/{numeric_id}/locations/{location}`
      `organizations/{numeric_id}/locations/{location}` where the location is
      for the Policy.
    policyId: Required. The ID to use for this policy, which will become the
      final component of the policy's resource name. The ID must contain 3 to
      63 characters. It can contain lowercase letters and numbers, as well as
      dashes (`-`) and periods (`.`). The first character must be a lowercase
      letter.
    validateOnly: Optional. If set to true, the request is validated and the
      user is provided with expected result, but no actual change is made.
  """
    googleIamV3betaV3Policy = _messages.MessageField('GoogleIamV3betaV3Policy', 1)
    parent = _messages.StringField(2, required=True)
    policyId = _messages.StringField(3)
    validateOnly = _messages.BooleanField(4)