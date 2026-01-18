from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ConnectorsProjectsLocationsProvidersConnectorsVersionsGetRequest(_messages.Message):
    """A ConnectorsProjectsLocationsProvidersConnectorsVersionsGetRequest
  object.

  Enums:
    ViewValueValuesEnum: Specifies which fields of the ConnectorVersion are
      returned in the response. Defaults to `CUSTOMER` view.

  Fields:
    name: Required. Resource name of the form:
      `projects/*/locations/*/providers/*/connectors/*/versions/*` Only global
      location is supported for ConnectorVersion resource.
    view: Specifies which fields of the ConnectorVersion are returned in the
      response. Defaults to `CUSTOMER` view.
  """

    class ViewValueValuesEnum(_messages.Enum):
        """Specifies which fields of the ConnectorVersion are returned in the
    response. Defaults to `CUSTOMER` view.

    Values:
      CONNECTOR_VERSION_VIEW_UNSPECIFIED: CONNECTOR_VERSION_VIEW_UNSPECIFIED.
      CONNECTOR_VERSION_VIEW_BASIC: Do not include role grant configs.
      CONNECTOR_VERSION_VIEW_FULL: Include role grant configs.
    """
        CONNECTOR_VERSION_VIEW_UNSPECIFIED = 0
        CONNECTOR_VERSION_VIEW_BASIC = 1
        CONNECTOR_VERSION_VIEW_FULL = 2
    name = _messages.StringField(1, required=True)
    view = _messages.EnumField('ViewValueValuesEnum', 2)