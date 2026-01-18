from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CancelWipeDeviceRequest(_messages.Message):
    """Request message for cancelling an unfinished device wipe.

  Fields:
    customer: Optional. [Resource
      name](https://cloud.google.com/apis/design/resource_names) of the
      customer. If you're using this API for your own organization, use
      `customers/my_customer` If you're using this API to manage another
      organization, use `customers/{customer_id}`, where customer_id is the
      customer to whom the device belongs.
  """
    customer = _messages.StringField(1)