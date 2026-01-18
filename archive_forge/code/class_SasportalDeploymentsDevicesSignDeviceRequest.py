from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SasportalDeploymentsDevicesSignDeviceRequest(_messages.Message):
    """A SasportalDeploymentsDevicesSignDeviceRequest object.

  Fields:
    name: Output only. The resource path name.
    sasPortalSignDeviceRequest: A SasPortalSignDeviceRequest resource to be
      passed as the request body.
  """
    name = _messages.StringField(1, required=True)
    sasPortalSignDeviceRequest = _messages.MessageField('SasPortalSignDeviceRequest', 2)