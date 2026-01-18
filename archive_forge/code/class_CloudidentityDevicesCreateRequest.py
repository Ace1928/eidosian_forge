from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudidentityDevicesCreateRequest(_messages.Message):
    """A CloudidentityDevicesCreateRequest object.

  Fields:
    customer: Optional. [Resource
      name](https://cloud.google.com/apis/design/resource_names) of the
      customer. If you're using this API for your own organization, use
      `customers/my_customer` If you're using this API to manage another
      organization, use `customers/{customer}`, where customer is the customer
      to whom the device belongs.
    googleAppsCloudidentityDevicesV1Device: A
      GoogleAppsCloudidentityDevicesV1Device resource to be passed as the
      request body.
  """
    customer = _messages.StringField(1)
    googleAppsCloudidentityDevicesV1Device = _messages.MessageField('GoogleAppsCloudidentityDevicesV1Device', 2)