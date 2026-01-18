from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IamOrganizationsLocationsPrincipalAccessBoundaryPoliciesGetRequest(_messages.Message):
    """A IamOrganizationsLocationsPrincipalAccessBoundaryPoliciesGetRequest
  object.

  Fields:
    name: Required. The name of the principal access boundary policy to
      retrieve. Format: `organizations/{organization_id}/locations/{location}/
      principalAccessBoundaryPolicies/{principal_access_boundary_policy_id}`
  """
    name = _messages.StringField(1, required=True)