from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ResourcesettingsProjectsSettingsPatchRequest(_messages.Message):
    """A ResourcesettingsProjectsSettingsPatchRequest object.

  Fields:
    googleCloudResourcesettingsV1Setting: A
      GoogleCloudResourcesettingsV1Setting resource to be passed as the
      request body.
    name: The resource name of the setting. Must be in one of the following
      forms: * `projects/{project_number}/settings/{setting_name}` *
      `folders/{folder_id}/settings/{setting_name}` *
      `organizations/{organization_id}/settings/{setting_name}` For example,
      "/projects/123/settings/gcp-enableMyFeature"
  """
    googleCloudResourcesettingsV1Setting = _messages.MessageField('GoogleCloudResourcesettingsV1Setting', 1)
    name = _messages.StringField(2, required=True)