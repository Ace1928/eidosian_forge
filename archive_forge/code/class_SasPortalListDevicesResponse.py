from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SasPortalListDevicesResponse(_messages.Message):
    """Response for ListDevices.

  Fields:
    devices: The devices that match the request.
    nextPageToken: A pagination token returned from a previous call to
      ListDevices that indicates from where listing should continue. If the
      field is missing or empty, it means there is no more devices.
  """
    devices = _messages.MessageField('SasPortalDevice', 1, repeated=True)
    nextPageToken = _messages.StringField(2)