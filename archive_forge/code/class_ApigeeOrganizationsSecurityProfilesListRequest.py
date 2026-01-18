from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsSecurityProfilesListRequest(_messages.Message):
    """A ApigeeOrganizationsSecurityProfilesListRequest object.

  Fields:
    pageSize: The maximum number of profiles to return. The service may return
      fewer than this value. If unspecified, at most 50 profiles will be
      returned.
    pageToken: A page token, received from a previous `ListSecurityProfiles`
      call. Provide this to retrieve the subsequent page.
    parent: Required. For a specific organization, list of all the security
      profiles. Format: `organizations/{org}`
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)