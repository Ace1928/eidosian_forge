from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsSecurityProfilesV2PatchRequest(_messages.Message):
    """A ApigeeOrganizationsSecurityProfilesV2PatchRequest object.

  Fields:
    googleCloudApigeeV1SecurityProfileV2: A
      GoogleCloudApigeeV1SecurityProfileV2 resource to be passed as the
      request body.
    name: Identifier. Name of the security profile v2 resource. Format:
      organizations/{org}/securityProfilesV2/{profile}
    updateMask: Required. The list of fields to update.
  """
    googleCloudApigeeV1SecurityProfileV2 = _messages.MessageField('GoogleCloudApigeeV1SecurityProfileV2', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)