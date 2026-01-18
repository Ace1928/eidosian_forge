from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecuritycenterFoldersLocationsMuteConfigsCreateRequest(_messages.Message):
    """A SecuritycenterFoldersLocationsMuteConfigsCreateRequest object.

  Fields:
    googleCloudSecuritycenterV2MuteConfig: A
      GoogleCloudSecuritycenterV2MuteConfig resource to be passed as the
      request body.
    muteConfigId: Required. Unique identifier provided by the client within
      the parent scope. It must consist of only lowercase letters, numbers,
      and hyphens, must start with a letter, must end with either a letter or
      a number, and must be 63 characters or less.
    parent: Required. Resource name of the new mute configs's parent. Its
      format is "organizations/[organization_id]/locations/[location_id]",
      "folders/[folder_id]/locations/[location_id]", or
      "projects/[project_id]/locations/[location_id]".
  """
    googleCloudSecuritycenterV2MuteConfig = _messages.MessageField('GoogleCloudSecuritycenterV2MuteConfig', 1)
    muteConfigId = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)