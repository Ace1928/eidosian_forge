from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DirectoryMobiledevicesListRequest(_messages.Message):
    """A DirectoryMobiledevicesListRequest object.

  Enums:
    OrderByValueValuesEnum: Column to use for sorting results
    ProjectionValueValuesEnum: Restrict information returned to a set of
      selected fields.
    SortOrderValueValuesEnum: Whether to return results in ascending or
      descending order. Only of use when orderBy is also used

  Fields:
    customerId: Immutable ID of the G Suite account
    maxResults: Maximum number of results to return. Max allowed value is 100.
    orderBy: Column to use for sorting results
    pageToken: Token to specify next page in the list
    projection: Restrict information returned to a set of selected fields.
    query: Search string in the format given at
      http://support.google.com/a/bin/answer.py?answer=1408863#search
    sortOrder: Whether to return results in ascending or descending order.
      Only of use when orderBy is also used
  """

    class OrderByValueValuesEnum(_messages.Enum):
        """Column to use for sorting results

    Values:
      deviceId: Mobile Device serial number.
      email: Owner user email.
      lastSync: Last policy settings sync date time of the device.
      model: Mobile Device model.
      name: Owner user name.
      os: Mobile operating system.
      status: Status of the device.
      type: Type of the device.
    """
        deviceId = 0
        email = 1
        lastSync = 2
        model = 3
        name = 4
        os = 5
        status = 6
        type = 7

    class ProjectionValueValuesEnum(_messages.Enum):
        """Restrict information returned to a set of selected fields.

    Values:
      BASIC: Includes only the basic metadata fields (e.g., deviceId, model,
        status, type, and status)
      FULL: Includes all metadata fields
    """
        BASIC = 0
        FULL = 1

    class SortOrderValueValuesEnum(_messages.Enum):
        """Whether to return results in ascending or descending order.

    Only of
    use when orderBy is also used

    Values:
      ASCENDING: Ascending order.
      DESCENDING: Descending order.
    """
        ASCENDING = 0
        DESCENDING = 1
    customerId = _messages.StringField(1, required=True)
    maxResults = _messages.IntegerField(2, variant=_messages.Variant.INT32, default=100)
    orderBy = _messages.EnumField('OrderByValueValuesEnum', 3)
    pageToken = _messages.StringField(4)
    projection = _messages.EnumField('ProjectionValueValuesEnum', 5)
    query = _messages.StringField(6)
    sortOrder = _messages.EnumField('SortOrderValueValuesEnum', 7)