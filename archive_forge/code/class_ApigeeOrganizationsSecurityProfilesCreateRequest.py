from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsSecurityProfilesCreateRequest(_messages.Message):
    """A ApigeeOrganizationsSecurityProfilesCreateRequest object.

  Fields:
    googleCloudApigeeV1SecurityProfile: A GoogleCloudApigeeV1SecurityProfile
      resource to be passed as the request body.
    parent: Required. Name of organization. Format: organizations/{org}
    securityProfileId: Required. The ID to use for the SecurityProfile, which
      will become the final component of the action's resource name. This
      value should be 1-63 characters and validated by
      "(^[a-z]([a-z0-9-]{0,61}[a-z0-9])?$)".
  """
    googleCloudApigeeV1SecurityProfile = _messages.MessageField('GoogleCloudApigeeV1SecurityProfile', 1)
    parent = _messages.StringField(2, required=True)
    securityProfileId = _messages.StringField(3)