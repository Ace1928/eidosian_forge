from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ResourcesettingsOrganizationsSettingsGetValueRequest(_messages.Message):
    """A ResourcesettingsOrganizationsSettingsGetValueRequest object.

  Fields:
    name: The name of the setting value to get. See SettingValue for naming
      requirements.
  """
    name = _messages.StringField(1, required=True)