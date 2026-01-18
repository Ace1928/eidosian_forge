from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworksecurityProjectsLocationsSecurityProfileGroupsCreateRequest(_messages.Message):
    """A NetworksecurityProjectsLocationsSecurityProfileGroupsCreateRequest
  object.

  Fields:
    parent: Required. The parent resource of the SecurityProfileGroup. Must be
      in the format `projects|organizations/*/locations/{location}`.
    securityProfileGroup: A SecurityProfileGroup resource to be passed as the
      request body.
    securityProfileGroupId: Required. Short name of the SecurityProfileGroup
      resource to be created. This value should be 1-63 characters long,
      containing only letters, numbers, hyphens, and underscores, and should
      not start with a number. E.g. "security_profile_group1".
  """
    parent = _messages.StringField(1, required=True)
    securityProfileGroup = _messages.MessageField('SecurityProfileGroup', 2)
    securityProfileGroupId = _messages.StringField(3)