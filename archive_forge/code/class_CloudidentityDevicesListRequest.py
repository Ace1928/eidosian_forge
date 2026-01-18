from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudidentityDevicesListRequest(_messages.Message):
    """A CloudidentityDevicesListRequest object.

  Enums:
    ViewValueValuesEnum: Optional. The view to use for the List request.

  Fields:
    customer: Optional. [Resource
      name](https://cloud.google.com/apis/design/resource_names) of the
      customer in the format: `customers/{customer}`, where customer is the
      customer to whom the device belongs. If you're using this API for your
      own organization, use `customers/my_customer`. If you're using this API
      to manage another organization, use `customers/{customer}`, where
      customer is the customer to whom the device belongs.
    filter: Optional. Additional restrictions when fetching list of devices.
      For a list of search fields, refer to [Mobile device search
      fields](https://developers.google.com/admin-sdk/directory/v1/search-
      operators). Multiple search fields are separated by the space character.
    orderBy: Optional. Order specification for devices in the response. Only
      one of the following field names may be used to specify the order:
      `create_time`, `last_sync_time`, `model`, `os_version`, `device_type`
      and `serial_number`. `desc` may be specified optionally at the end to
      specify results to be sorted in descending order. Default order is
      ascending.
    pageSize: Optional. The maximum number of Devices to return. If
      unspecified, at most 20 Devices will be returned. The maximum value is
      100; values above 100 will be coerced to 100.
    pageToken: Optional. A page token, received from a previous `ListDevices`
      call. Provide this to retrieve the subsequent page. When paginating, all
      other parameters provided to `ListDevices` must match the call that
      provided the page token.
    view: Optional. The view to use for the List request.
  """

    class ViewValueValuesEnum(_messages.Enum):
        """Optional. The view to use for the List request.

    Values:
      VIEW_UNSPECIFIED: Default value. The value is unused.
      COMPANY_INVENTORY: This view contains all devices imported by the
        company admin. Each device in the response contains all information
        specified by the company admin when importing the device (i.e. asset
        tags). This includes devices that may be unaassigned or assigned to
        users.
      USER_ASSIGNED_DEVICES: This view contains all devices with at least one
        user registered on the device. Each device in the response contains
        all device information, except for asset tags.
    """
        VIEW_UNSPECIFIED = 0
        COMPANY_INVENTORY = 1
        USER_ASSIGNED_DEVICES = 2
    customer = _messages.StringField(1)
    filter = _messages.StringField(2)
    orderBy = _messages.StringField(3)
    pageSize = _messages.IntegerField(4, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(5)
    view = _messages.EnumField('ViewValueValuesEnum', 6)