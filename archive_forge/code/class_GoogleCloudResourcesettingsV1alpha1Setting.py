from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class GoogleCloudResourcesettingsV1alpha1Setting(_messages.Message):
    """The schema for setting values. At a given Cloud resource, a setting can
  parent at most one setting value.

  Enums:
    DataTypeValueValuesEnum: The data type for this setting.

  Fields:
    dataType: The data type for this setting.
    defaultValue: The value received by LookupEffectiveSettingValue if no
      setting value is explicitly set. Note: not all settings have a default
      value.
    description: A detailed description of what this setting does.
    displayName: The human readable name for this setting.
    name: The resource name of the setting. Must be in one of the following
      forms: * `projects/{project_number}/settings/{setting_name}` *
      `folders/{folder_id}/settings/{setting_name}` *
      `organizations/{organization_id}/settings/{setting_name}` For example,
      "/projects/123/settings/gcp-enableMyFeature"
    readOnly: A flag indicating that values of this setting cannot be modified
      (see documentation of the specific setting for updates and reasons);
      however, it may be deleted using DeleteSettingValue if
      DeleteSettingValueRequest.ignore_read_only is set to true. Using this
      flag is considered an acknowledgement that the setting value cannot be
      recreated. See DeleteSettingValueRequest.ignore_read_only for more
      details.
  """

    class DataTypeValueValuesEnum(_messages.Enum):
        """The data type for this setting.

    Values:
      DATA_TYPE_UNSPECIFIED: Unspecified data type.
      BOOLEAN: A boolean setting.
      STRING: A string setting.
      STRING_SET: A string set setting.
      ENUM_VALUE: A Enum setting
      DURATION_VALUE: A Duration setting
      STRING_MAP: A string->string map setting
    """
        DATA_TYPE_UNSPECIFIED = 0
        BOOLEAN = 1
        STRING = 2
        STRING_SET = 3
        ENUM_VALUE = 4
        DURATION_VALUE = 5
        STRING_MAP = 6
    dataType = _messages.EnumField('DataTypeValueValuesEnum', 1)
    defaultValue = _messages.MessageField('GoogleCloudResourcesettingsV1alpha1Value', 2)
    description = _messages.StringField(3)
    displayName = _messages.StringField(4)
    name = _messages.StringField(5)
    readOnly = _messages.BooleanField(6)