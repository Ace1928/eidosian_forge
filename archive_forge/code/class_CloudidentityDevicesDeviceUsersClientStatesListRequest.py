from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudidentityDevicesDeviceUsersClientStatesListRequest(_messages.Message):
    """A CloudidentityDevicesDeviceUsersClientStatesListRequest object.

  Fields:
    customer: Optional. [Resource
      name](https://cloud.google.com/apis/design/resource_names) of the
      customer. If you're using this API for your own organization, use
      `customers/my_customer` If you're using this API to manage another
      organization, use `customers/{customer}`, where customer is the customer
      to whom the device belongs.
    filter: Optional. Additional restrictions when fetching list of client
      states.
    orderBy: Optional. Order specification for client states in the response.
    pageToken: Optional. A page token, received from a previous
      `ListClientStates` call. Provide this to retrieve the subsequent page.
      When paginating, all other parameters provided to `ListClientStates`
      must match the call that provided the page token.
    parent: Required. To list all ClientStates, set this to
      "devices/-/deviceUsers/-". To list all ClientStates owned by a
      DeviceUser, set this to the resource name of the DeviceUser. Format:
      devices/{device}/deviceUsers/{deviceUser}
  """
    customer = _messages.StringField(1)
    filter = _messages.StringField(2)
    orderBy = _messages.StringField(3)
    pageToken = _messages.StringField(4)
    parent = _messages.StringField(5, required=True)