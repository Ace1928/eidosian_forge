from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudidentityDevicesGetRequest(_messages.Message):
    """A CloudidentityDevicesGetRequest object.

  Fields:
    customer: Optional. [Resource
      name](https://cloud.google.com/apis/design/resource_names) of the
      Customer in the format: `customers/{customer}`, where customer is the
      customer to whom the device belongs. If you're using this API for your
      own organization, use `customers/my_customer`. If you're using this API
      to manage another organization, use `customers/{customer}`, where
      customer is the customer to whom the device belongs.
    name: Required. [Resource
      name](https://cloud.google.com/apis/design/resource_names) of the Device
      in the format: `devices/{device}`, where device is the unique ID
      assigned to the Device.
  """
    customer = _messages.StringField(1)
    name = _messages.StringField(2, required=True)