from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IamOrganizationsLocationsPrincipalAccessBoundaryPoliciesPatchRequest(_messages.Message):
    """A IamOrganizationsLocationsPrincipalAccessBoundaryPoliciesPatchRequest
  object.

  Fields:
    googleIamV3betaPrincipalAccessBoundaryPolicy: A
      GoogleIamV3betaPrincipalAccessBoundaryPolicy resource to be passed as
      the request body.
    name: Identifier. The resource name of the principal access boundary
      policy. The following format is supported: `organizations/{organization_
      id}/locations/{location}/principalAccessBoundaryPolicies/{policy_id}`
    updateMask: Optional. The list of fields to update
    validateOnly: Optional. If set, validate the request and preview the
      update, but do not actually post it.
  """
    googleIamV3betaPrincipalAccessBoundaryPolicy = _messages.MessageField('GoogleIamV3betaPrincipalAccessBoundaryPolicy', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)
    validateOnly = _messages.BooleanField(4)