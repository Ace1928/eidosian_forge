from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class DnsManagedZoneOperationsListRequest(_messages.Message):
    """A DnsManagedZoneOperationsListRequest object.

  Enums:
    SortByValueValuesEnum: Sorting criterion. The only supported values are
      START_TIME and ID.

  Fields:
    managedZone: Identifies the managed zone addressed by this request.
    maxResults: Optional. Maximum number of results to be returned. If
      unspecified, the server decides how many results to return.
    pageToken: Optional. A tag returned by a previous list request that was
      truncated. Use this parameter to continue a previous list request.
    project: Identifies the project addressed by this request.
    sortBy: Sorting criterion. The only supported values are START_TIME and
      ID.
  """

    class SortByValueValuesEnum(_messages.Enum):
        """Sorting criterion. The only supported values are START_TIME and ID.

    Values:
      startTime: <no description>
      id: <no description>
    """
        startTime = 0
        id = 1
    managedZone = _messages.StringField(1, required=True)
    maxResults = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    project = _messages.StringField(4, required=True)
    sortBy = _messages.EnumField('SortByValueValuesEnum', 5, default='startTime')