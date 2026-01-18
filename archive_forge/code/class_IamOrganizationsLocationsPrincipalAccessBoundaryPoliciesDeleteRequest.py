from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IamOrganizationsLocationsPrincipalAccessBoundaryPoliciesDeleteRequest(_messages.Message):
    """A IamOrganizationsLocationsPrincipalAccessBoundaryPoliciesDeleteRequest
  object.

  Fields:
    etag: Optional. The etag of the principal access boundary policy. If this
      is provided, it must match the server's etag.
    force: Optional. If set to true, the request will force the deletion of
      the Policy even if the Policy references PolicyBindings.
    name: Required. The name of the principal access boundary policy to
      delete. Format: `organizations/{organization_id}/locations/{location}/pr
      incipalAccessBoundaryPolicies/{principal_access_boundary_policy_id}`
    validateOnly: Optional. If set, validate the request and preview the
      deletion, but do not actually post it.
  """
    etag = _messages.StringField(1)
    force = _messages.BooleanField(2)
    name = _messages.StringField(3, required=True)
    validateOnly = _messages.BooleanField(4)