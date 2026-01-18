from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ResourcesettingsOrganizationsSettingsGetRequest(_messages.Message):
    """A ResourcesettingsOrganizationsSettingsGetRequest object.

  Enums:
    ViewValueValuesEnum: The SettingView for this request.

  Fields:
    name: Required. The name of the setting to get. See Setting for naming
      requirements.
    view: The SettingView for this request.
  """

    class ViewValueValuesEnum(_messages.Enum):
        """The SettingView for this request.

    Values:
      SETTING_VIEW_UNSPECIFIED: The default / unset value. The API will
        default to the SETTING_VIEW_BASIC view.
      SETTING_VIEW_BASIC: Include Setting.metadata, but nothing else. This is
        the default value (for both ListSettings and GetSetting).
      SETTING_VIEW_EFFECTIVE_VALUE: Include Setting.effective_value, but
        nothing else.
      SETTING_VIEW_LOCAL_VALUE: Include Setting.local_value, but nothing else.
    """
        SETTING_VIEW_UNSPECIFIED = 0
        SETTING_VIEW_BASIC = 1
        SETTING_VIEW_EFFECTIVE_VALUE = 2
        SETTING_VIEW_LOCAL_VALUE = 3
    name = _messages.StringField(1, required=True)
    view = _messages.EnumField('ViewValueValuesEnum', 2)