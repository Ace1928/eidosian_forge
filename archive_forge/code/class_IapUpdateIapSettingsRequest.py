from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IapUpdateIapSettingsRequest(_messages.Message):
    """A IapUpdateIapSettingsRequest object.

  Fields:
    iapSettings: A IapSettings resource to be passed as the request body.
    name: Required. The resource name of the IAP protected resource.
    updateMask: The field mask specifying which IAP settings should be
      updated. If omitted, then all of the settings are updated. See
      https://developers.google.com/protocol-
      buffers/docs/reference/google.protobuf#fieldmask. Note: All IAP reauth
      settings must always be set together, using the field mask:
      `iapSettings.accessSettings.reauthSettings`.
  """
    iapSettings = _messages.MessageField('IapSettings', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)