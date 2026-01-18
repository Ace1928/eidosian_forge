from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworksecurityProjectsLocationsSecurityProfilesDeleteRequest(_messages.Message):
    """A NetworksecurityProjectsLocationsSecurityProfilesDeleteRequest object.

  Fields:
    etag: Optional. If client provided etag is out of date, delete will return
      FAILED_PRECONDITION error.
    name: Required. A name of the SecurityProfile to delete. Must be in the
      format `projects|organizations/*/locations/{location}/securityProfiles/{
      security_profile_id}`.
  """
    etag = _messages.StringField(1)
    name = _messages.StringField(2, required=True)