from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PrivilegedaccessmanagerOrganizationsLocationsEntitlementsCreateRequest(_messages.Message):
    """A PrivilegedaccessmanagerOrganizationsLocationsEntitlementsCreateRequest
  object.

  Fields:
    entitlement: A Entitlement resource to be passed as the request body.
    entitlementId: Required. The ID to use for this Entitlement. This will
      become the last part of the resource name. This value should be 4-63
      characters, and valid characters are "[a-z]", "[0-9]", and "-". The
      first character should be from [a-z]. This value should be unique among
      all other Entitlements under the specified `parent`.
    parent: Required. Name of the parent resource for the to-be-created
      Entitlement. Possible formats: * `organizations/{organization-
      number}/locations/{region}` * `folders/{folder-
      number}/locations/{region}` * `projects/{project-id|project-
      number}/locations/{region}`
    requestId: Optional. An optional request ID to identify requests. Specify
      a unique request ID so that if you must retry your request, the server
      will know to ignore the request if it has already been completed. The
      server will guarantee that for at least 60 minutes since the first
      request. For example, consider a situation where you make an initial
      request and the request times out. If you make the request again with
      the same request ID, the server can check if original operation with the
      same request ID was received, and if so, will ignore the second request
      and return the previous operations response. This prevents clients from
      accidentally creating duplicate commitments. The request ID must be a
      valid UUID with the exception that zero UUID is not supported
      (00000000-0000-0000-0000-000000000000).
  """
    entitlement = _messages.MessageField('Entitlement', 1)
    entitlementId = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)
    requestId = _messages.StringField(4)