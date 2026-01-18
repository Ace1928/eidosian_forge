from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ConnectorsProjectsLocationsConnectionsListRequest(_messages.Message):
    """A ConnectorsProjectsLocationsConnectionsListRequest object.

  Enums:
    ViewValueValuesEnum: Specifies which fields of the Connection are returned
      in the response. Defaults to `BASIC` view.

  Fields:
    filter: Filter.
    orderBy: Order by parameters.
    pageSize: Page size.
    pageToken: Page token.
    parent: Required. Parent resource of the Connection, of the form:
      `projects/*/locations/*`
    view: Specifies which fields of the Connection are returned in the
      response. Defaults to `BASIC` view.
  """

    class ViewValueValuesEnum(_messages.Enum):
        """Specifies which fields of the Connection are returned in the response.
    Defaults to `BASIC` view.

    Values:
      CONNECTION_VIEW_UNSPECIFIED: CONNECTION_UNSPECIFIED.
      BASIC: Do not include runtime required configs.
      FULL: Include runtime required configs.
    """
        CONNECTION_VIEW_UNSPECIFIED = 0
        BASIC = 1
        FULL = 2
    filter = _messages.StringField(1)
    orderBy = _messages.StringField(2)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)
    parent = _messages.StringField(5, required=True)
    view = _messages.EnumField('ViewValueValuesEnum', 6)