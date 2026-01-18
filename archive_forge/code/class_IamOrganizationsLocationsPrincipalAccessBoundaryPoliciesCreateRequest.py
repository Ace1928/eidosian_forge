from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IamOrganizationsLocationsPrincipalAccessBoundaryPoliciesCreateRequest(_messages.Message):
    """A IamOrganizationsLocationsPrincipalAccessBoundaryPoliciesCreateRequest
  object.

  Fields:
    googleIamV3betaPrincipalAccessBoundaryPolicy: A
      GoogleIamV3betaPrincipalAccessBoundaryPolicy resource to be passed as
      the request body.
    parent: Required. The parent resource where this principal access boundary
      policy will be created. Only organization is supported now. Format:
      `organizations/{organization_id}/locations/{location}`
    principalAccessBoundaryPolicyId: Required. The ID to use for the principal
      access boundary policy, which will become the final component of the
      principal access boundary policy's resource name. This value must start
      with a lowercase letter followed by up to 62 lowercase letters, numbers,
      hyphens, or dots. Pattern, /a-z{2,62}/.
    validateOnly: Optional. If set, validate the request and preview the
      creation, but do not actually post it.
  """
    googleIamV3betaPrincipalAccessBoundaryPolicy = _messages.MessageField('GoogleIamV3betaPrincipalAccessBoundaryPolicy', 1)
    parent = _messages.StringField(2, required=True)
    principalAccessBoundaryPolicyId = _messages.StringField(3)
    validateOnly = _messages.BooleanField(4)