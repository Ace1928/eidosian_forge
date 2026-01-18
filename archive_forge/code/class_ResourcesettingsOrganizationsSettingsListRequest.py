from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ResourcesettingsOrganizationsSettingsListRequest(_messages.Message):
    """A ResourcesettingsOrganizationsSettingsListRequest object.

  Enums:
    ViewValueValuesEnum: The SettingView for this request.

  Fields:
    pageSize: Unused. The size of the page to be returned.
    pageToken: Unused. A page token used to retrieve the next page.
    parent: Required. The project, folder, or organization that is the parent
      resource for this setting. Must be in one of the following forms: *
      `projects/{project_number}` * `projects/{project_id}` *
      `folders/{folder_id}` * `organizations/{organization_id}`
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
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)
    view = _messages.EnumField('ViewValueValuesEnum', 4)