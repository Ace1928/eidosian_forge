from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LookupSelfDeviceUsersResponse(_messages.Message):
    """Response containing resource names of the DeviceUsers associated with
  the caller's credentials.

  Fields:
    customer: The customer Id that may be passed back to other Devices API
      methods such as List, Get, etc.
    names: [Resource
      names](https://cloud.google.com/apis/design/resource_names) of the
      DeviceUsers in the format:
      `devices/{device_id}/deviceUsers/{user_resource_id}`, where device_id is
      the unique ID assigned to a Device and user_resource_id is the unique
      user ID
    nextPageToken: Token to retrieve the next page of results. Empty if there
      are no more results.
  """
    customer = _messages.StringField(1)
    names = _messages.StringField(2, repeated=True)
    nextPageToken = _messages.StringField(3)