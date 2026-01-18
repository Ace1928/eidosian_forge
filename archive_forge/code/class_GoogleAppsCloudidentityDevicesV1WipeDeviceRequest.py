from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleAppsCloudidentityDevicesV1WipeDeviceRequest(_messages.Message):
    """Request message for wiping all data on the device.

  Fields:
    customer: Optional. [Resource
      name](https://cloud.google.com/apis/design/resource_names) of the
      customer. If you're using this API for your own organization, use
      `customers/my_customer` If you're using this API to manage another
      organization, use `customers/{customer}`, where customer is the customer
      to whom the device belongs.
    removeResetLock: Optional. Specifies if a user is able to factory reset a
      device after a Device Wipe. On iOS, this is called "Activation Lock",
      while on Android, this is known as "Factory Reset Protection". If true,
      this protection will be removed from the device, so that a user can
      successfully factory reset. If false, the setting is untouched on the
      device.
  """
    customer = _messages.StringField(1)
    removeResetLock = _messages.BooleanField(2)