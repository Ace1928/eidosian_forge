from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class GoogleCloudResourcesettingsV1alpha1ListSettingsResponse(_messages.Message):
    """The response from ListSettings.

  Fields:
    nextPageToken: Unused. A page token used to retrieve the next page.
    settings: A list of settings that are available at the specified Cloud
      resource.
  """
    nextPageToken = _messages.StringField(1)
    settings = _messages.MessageField('GoogleCloudResourcesettingsV1alpha1Setting', 2, repeated=True)