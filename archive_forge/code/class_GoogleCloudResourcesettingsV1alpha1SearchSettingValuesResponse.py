from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class GoogleCloudResourcesettingsV1alpha1SearchSettingValuesResponse(_messages.Message):
    """The response from SearchSettingValues.

  Fields:
    nextPageToken: Unused. A page token used to retrieve the next page.
    settingValues: All setting values that exist on the specified Cloud
      resource.
  """
    nextPageToken = _messages.StringField(1)
    settingValues = _messages.MessageField('GoogleCloudResourcesettingsV1alpha1SettingValue', 2, repeated=True)