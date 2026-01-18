from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworksecurityProjectsLocationsSecurityProfilesGetRequest(_messages.Message):
    """A NetworksecurityProjectsLocationsSecurityProfilesGetRequest object.

  Fields:
    name: Required. A name of the SecurityProfile to get. Must be in the
      format `projects|organizations/*/locations/{location}/securityProfiles/{
      security_profile_id}`.
  """
    name = _messages.StringField(1, required=True)