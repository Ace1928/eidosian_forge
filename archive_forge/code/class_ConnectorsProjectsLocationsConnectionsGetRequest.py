from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ConnectorsProjectsLocationsConnectionsGetRequest(_messages.Message):
    """A ConnectorsProjectsLocationsConnectionsGetRequest object.

  Enums:
    ViewValueValuesEnum: Specifies which fields of the Connection are returned
      in the response. Defaults to `BASIC` view.

  Fields:
    name: Required. Resource name of the form:
      `projects/*/locations/*/connections/*`
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
    name = _messages.StringField(1, required=True)
    view = _messages.EnumField('ViewValueValuesEnum', 2)