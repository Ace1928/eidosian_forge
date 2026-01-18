from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SasportalDeploymentsDevicesMoveRequest(_messages.Message):
    """A SasportalDeploymentsDevicesMoveRequest object.

  Fields:
    name: Required. The name of the device to move.
    sasPortalMoveDeviceRequest: A SasPortalMoveDeviceRequest resource to be
      passed as the request body.
  """
    name = _messages.StringField(1, required=True)
    sasPortalMoveDeviceRequest = _messages.MessageField('SasPortalMoveDeviceRequest', 2)