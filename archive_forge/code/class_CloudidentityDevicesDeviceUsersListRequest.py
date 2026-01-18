from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudidentityDevicesDeviceUsersListRequest(_messages.Message):
    """A CloudidentityDevicesDeviceUsersListRequest object.

  Fields:
    customer: Optional. [Resource
      name](https://cloud.google.com/apis/design/resource_names) of the
      customer. If you're using this API for your own organization, use
      `customers/my_customer` If you're using this API to manage another
      organization, use `customers/{customer}`, where customer is the customer
      to whom the device belongs.
    filter: Optional. Additional restrictions when fetching list of devices.
      For a list of search fields, refer to [Mobile device search
      fields](https://developers.google.com/admin-sdk/directory/v1/search-
      operators). Multiple search fields are separated by the space character.
    orderBy: Optional. Order specification for devices in the response.
    pageSize: Optional. The maximum number of DeviceUsers to return. If
      unspecified, at most 5 DeviceUsers will be returned. The maximum value
      is 20; values above 20 will be coerced to 20.
    pageToken: Optional. A page token, received from a previous
      `ListDeviceUsers` call. Provide this to retrieve the subsequent page.
      When paginating, all other parameters provided to `ListBooks` must match
      the call that provided the page token.
    parent: Required. To list all DeviceUsers, set this to "devices/-". To
      list all DeviceUsers owned by a device, set this to the resource name of
      the device. Format: devices/{device}
  """
    customer = _messages.StringField(1)
    filter = _messages.StringField(2)
    orderBy = _messages.StringField(3)
    pageSize = _messages.IntegerField(4, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(5)
    parent = _messages.StringField(6, required=True)