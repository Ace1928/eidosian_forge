from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IamOrganizationsLocationsPrincipalAccessBoundaryPoliciesSearchPolicyBindingsRequest(_messages.Message):
    """A IamOrganizationsLocationsPrincipalAccessBoundaryPoliciesSearchPolicyBi
  ndingsRequest object.

  Fields:
    name: Required. The name of the principal access boundary policy. Format:
      `organizations/{organization_id}/locations/{location}/principalAccessBou
      ndaryPolicies/{principal_access_boundary_policy_id}`
    pageSize: Optional. The maximum number of policy bindings to return. The
      service may return fewer than this value. If unspecified, at most 50
      policy bindings will be returned. The maximum value is 1000; values
      above 1000 will be coerced to 1000.
    pageToken: Optional. A page token, received from a previous
      `SearchPrincipalAccessBoundaryPolicyBindingsRequest` call. Provide this
      to retrieve the subsequent page. When paginating, all other parameters
      provided to `SearchPrincipalAccessBoundaryPolicyBindingsRequest` must
      match the call that provided the page token.
  """
    name = _messages.StringField(1, required=True)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)