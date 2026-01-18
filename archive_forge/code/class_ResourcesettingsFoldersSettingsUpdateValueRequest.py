from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ResourcesettingsFoldersSettingsUpdateValueRequest(_messages.Message):
    """A ResourcesettingsFoldersSettingsUpdateValueRequest object.

  Fields:
    googleCloudResourcesettingsV1alpha1SettingValue: A
      GoogleCloudResourcesettingsV1alpha1SettingValue resource to be passed as
      the request body.
    name: The resource name of the setting value. Must be in one of the
      following forms: *
      `projects/{project_number}/settings/{setting_name}/value` *
      `folders/{folder_id}/settings/{setting_name}/value` *
      `organizations/{organization_id}/settings/{setting_name}/value` For
      example, "/projects/123/settings/gcp-enableMyFeature/value"
  """
    googleCloudResourcesettingsV1alpha1SettingValue = _messages.MessageField('GoogleCloudResourcesettingsV1alpha1SettingValue', 1)
    name = _messages.StringField(2, required=True)