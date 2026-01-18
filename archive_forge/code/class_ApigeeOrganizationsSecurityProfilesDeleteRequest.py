from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsSecurityProfilesDeleteRequest(_messages.Message):
    """A ApigeeOrganizationsSecurityProfilesDeleteRequest object.

  Fields:
    name: Required. Name of profile. Format:
      organizations/{org}/securityProfiles/{profile}
  """
    name = _messages.StringField(1, required=True)