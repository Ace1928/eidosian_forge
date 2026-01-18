from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DirectoryChromeosdevicesListRequest(_messages.Message):
    """A DirectoryChromeosdevicesListRequest object.

  Enums:
    OrderByValueValuesEnum: Column to use for sorting results
    ProjectionValueValuesEnum: Restrict information returned to a set of
      selected fields.
    SortOrderValueValuesEnum: Whether to return results in ascending or
      descending order. Only of use when orderBy is also used

  Fields:
    customerId: Immutable ID of the G Suite account
    maxResults: Maximum number of results to return. Max allowed value is 200.
    orderBy: Column to use for sorting results
    orgUnitPath: Full path of the organizational unit or its ID
    pageToken: Token to specify next page in the list
    projection: Restrict information returned to a set of selected fields.
    query: Search string in the format given at
      http://support.google.com/chromeos/a/bin/answer.py?answer=1698333
    sortOrder: Whether to return results in ascending or descending order.
      Only of use when orderBy is also used
  """

    class OrderByValueValuesEnum(_messages.Enum):
        """Column to use for sorting results

    Values:
      annotatedLocation: Chromebook location as annotated by the
        administrator.
      annotatedUser: Chromebook user as annotated by administrator.
      lastSync: Chromebook last sync.
      notes: Chromebook notes as annotated by the administrator.
      serialNumber: Chromebook Serial Number.
      status: Chromebook status.
      supportEndDate: Chromebook support end date.
    """
        annotatedLocation = 0
        annotatedUser = 1
        lastSync = 2
        notes = 3
        serialNumber = 4
        status = 5
        supportEndDate = 6

    class ProjectionValueValuesEnum(_messages.Enum):
        """Restrict information returned to a set of selected fields.

    Values:
      BASIC: Includes only the basic metadata fields (e.g., deviceId,
        serialNumber, status, and user)
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
    orgUnitPath = _messages.StringField(4)
    pageToken = _messages.StringField(5)
    projection = _messages.EnumField('ProjectionValueValuesEnum', 6)
    query = _messages.StringField(7)
    sortOrder = _messages.EnumField('SortOrderValueValuesEnum', 8)