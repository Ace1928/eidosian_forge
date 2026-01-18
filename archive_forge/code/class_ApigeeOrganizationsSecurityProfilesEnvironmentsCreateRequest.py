from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsSecurityProfilesEnvironmentsCreateRequest(_messages.Message):
    """A ApigeeOrganizationsSecurityProfilesEnvironmentsCreateRequest object.

  Fields:
    googleCloudApigeeV1SecurityProfileEnvironmentAssociation: A
      GoogleCloudApigeeV1SecurityProfileEnvironmentAssociation resource to be
      passed as the request body.
    parent: Required. Name of organization and security profile ID. Format:
      organizations/{org}/securityProfiles/{profile}
  """
    googleCloudApigeeV1SecurityProfileEnvironmentAssociation = _messages.MessageField('GoogleCloudApigeeV1SecurityProfileEnvironmentAssociation', 1)
    parent = _messages.StringField(2, required=True)