from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SasPortalUpdateSignedDeviceRequest(_messages.Message):
    """Request for UpdateSignedDevice.

  Fields:
    encodedDevice: Required. The JSON Web Token signed using a CPI private
      key. Payload must be the JSON encoding of the device. The user_id field
      must be set.
    installerId: Required. Unique installer ID (CPI ID) from the Certified
      Professional Installers database.
  """
    encodedDevice = _messages.BytesField(1)
    installerId = _messages.StringField(2)