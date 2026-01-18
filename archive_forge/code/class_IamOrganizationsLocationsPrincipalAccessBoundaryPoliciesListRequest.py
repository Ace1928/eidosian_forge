from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IamOrganizationsLocationsPrincipalAccessBoundaryPoliciesListRequest(_messages.Message):
    """A IamOrganizationsLocationsPrincipalAccessBoundaryPoliciesListRequest
  object.

  Fields:
    pageSize: Optional. The maximum number of principal access boundary
      policies to return. The service may return fewer than this value. If
      unspecified, at most 50 principal access boundary policies will be
      returned. The maximum value is 1000; values above 1000 will be coerced
      to 1000.
    pageToken: Optional. A page token, received from a previous
      `ListPrincipalAccessBoundaryPolicies` call. Provide this to retrieve the
      subsequent page. When paginating, all other parameters provided to
      `ListPrincipalAccessBoundaryPolicies` must match the call that provided
      the page token.
    parent: Required. The parent resource, which owns the collection of
      principal access boundary policies. Format:
      `organizations/{organization_id}/locations/{location}`
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)