from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudidentityDevicesDeviceUsersDeleteRequest(_messages.Message):
    """A CloudidentityDevicesDeviceUsersDeleteRequest object.

  Fields:
    customer: Optional. [Resource
      name](https://cloud.google.com/apis/design/resource_names) of the
      customer. If you're using this API for your own organization, use
      `customers/my_customer` If you're using this API to manage another
      organization, use `customers/{customer}`, where customer is the customer
      to whom the device belongs.
    name: Required. [Resource
      name](https://cloud.google.com/apis/design/resource_names) of the Device
      in format: `devices/{device}/deviceUsers/{device_user}`, where device is
      the unique ID assigned to the Device, and device_user is the unique ID
      assigned to the User.
  """
    customer = _messages.StringField(1)
    name = _messages.StringField(2, required=True)