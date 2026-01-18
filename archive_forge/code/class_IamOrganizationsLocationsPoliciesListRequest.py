from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IamOrganizationsLocationsPoliciesListRequest(_messages.Message):
    """A IamOrganizationsLocationsPoliciesListRequest object.

  Fields:
    pageSize: Optional. Requested page size. Server may return fewer policies
      than requested. If unspecified, server will pick an appropriate default.
    pageToken: Optional. A token identifying a page of results the server
      should return. returned from the previous call to `ListPolicies` method.
    parent: Required. Parent value for ListPoliciesRequest. The parent needs
      to follow formats below. `projects/{project_id}/locations/{location}`
      `projects/{project_number}/locations/{location}`
      `folders/{numeric_id}/locations/{location}`
      `organizations/{numeric_id}/locations/{location}`
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)