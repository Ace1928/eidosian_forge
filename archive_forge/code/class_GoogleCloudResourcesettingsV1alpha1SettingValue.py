from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class GoogleCloudResourcesettingsV1alpha1SettingValue(_messages.Message):
    """The instantiation of a setting. Every setting value is parented by its
  corresponding setting.

  Fields:
    etag: A fingerprint used for optimistic concurrency. See
      UpdateSettingValue for more details.
    name: The resource name of the setting value. Must be in one of the
      following forms: *
      `projects/{project_number}/settings/{setting_name}/value` *
      `folders/{folder_id}/settings/{setting_name}/value` *
      `organizations/{organization_id}/settings/{setting_name}/value` For
      example, "/projects/123/settings/gcp-enableMyFeature/value"
    readOnly: Output only. A flag indicating that this setting value cannot be
      modified; however, it may be deleted using DeleteSettingValue if
      DeleteSettingValueRequest.ignore_read_only is set to true. Using this
      flag is considered an acknowledgement that the setting value cannot be
      recreated. This flag is inherited from its parent setting and is for
      convenience purposes. See Setting.read_only for more details.
    updateTime: Output only. The timestamp indicating when the setting value
      was last updated.
    value: The value of the setting. The data type of Value must always be
      consistent with the data type defined by the parent setting.
  """
    etag = _messages.StringField(1)
    name = _messages.StringField(2)
    readOnly = _messages.BooleanField(3)
    updateTime = _messages.StringField(4)
    value = _messages.MessageField('GoogleCloudResourcesettingsV1alpha1Value', 5)