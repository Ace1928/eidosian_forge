from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CreateDeviceRequest(_messages.Message):
    """Request message for creating a Company Owned device.

  Fields:
    customer: Optional. [Resource
      name](https://cloud.google.com/apis/design/resource_names) of the
      customer. If you're using this API for your own organization, use
      `customers/my_customer` If you're using this API to manage another
      organization, use `customers/{customer_id}`, where customer_id is the
      customer to whom the device belongs.
    device: Required. The device to be created. The name field within this
      device is ignored in the create method. A new name is created by the
      method, and returned within the response. Only the fields `device_type`,
      `serial_number` and `asset_tag` (if present) are used to create the
      device. All other fields are ignored. The `device_type` and
      `serial_number` fields are required.
  """
    customer = _messages.StringField(1)
    device = _messages.MessageField('Device', 2)