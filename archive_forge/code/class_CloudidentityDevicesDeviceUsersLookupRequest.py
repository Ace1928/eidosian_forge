from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudidentityDevicesDeviceUsersLookupRequest(_messages.Message):
    """A CloudidentityDevicesDeviceUsersLookupRequest object.

  Fields:
    androidId: Android Id returned by [Settings.Secure#ANDROID_ID](https://dev
      eloper.android.com/reference/android/provider/Settings.Secure.html#ANDRO
      ID_ID).
    pageSize: The maximum number of DeviceUsers to return. If unspecified, at
      most 20 DeviceUsers will be returned. The maximum value is 20; values
      above 20 will be coerced to 20.
    pageToken: A page token, received from a previous `LookupDeviceUsers`
      call. Provide this to retrieve the subsequent page. When paginating, all
      other parameters provided to `LookupDeviceUsers` must match the call
      that provided the page token.
    parent: Must be set to "devices/-/deviceUsers" to search across all
      DeviceUser belonging to the user.
    rawResourceId: Raw Resource Id used by Google Endpoint Verification. If
      the user is enrolled into Google Endpoint Verification, this id will be
      saved as the 'device_resource_id' field in the following platform
      dependent files. * macOS: ~/.secureConnect/context_aware_config.json *
      Windows: %USERPROFILE%\\AppData\\Local\\Google\\Endpoint
      Verification\\accounts.json * Linux:
      ~/.secureConnect/context_aware_config.json
    userId: The user whose DeviceUser's resource name will be fetched. Must be
      set to 'me' to fetch the DeviceUser's resource name for the calling
      user.
  """
    androidId = _messages.StringField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    parent = _messages.StringField(4, required=True)
    rawResourceId = _messages.StringField(5)
    userId = _messages.StringField(6)