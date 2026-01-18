from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NetworkconnectivityProjectsLocationsGlobalHubsListSpokesRequest(_messages.Message):
    """A NetworkconnectivityProjectsLocationsGlobalHubsListSpokesRequest
  object.

  Enums:
    ViewValueValuesEnum: The view of the spoke to return. The view that you
      use determines which spoke fields are included in the response.

  Fields:
    filter: An expression that filters the list of results.
    name: Required. The name of the hub.
    orderBy: Sort the results by name or create_time.
    pageSize: The maximum number of results to return per page.
    pageToken: The page token.
    spokeLocations: A list of locations. Specify one of the following:
      `[global]`, a single region (for example, `[us-central1]`), or a
      combination of values (for example, `[global, us-central1, us-west1]`).
      If the spoke_locations field is populated, the list of results includes
      only spokes in the specified location. If the spoke_locations field is
      not populated, the list of results includes spokes in all locations.
    view: The view of the spoke to return. The view that you use determines
      which spoke fields are included in the response.
  """

    class ViewValueValuesEnum(_messages.Enum):
        """The view of the spoke to return. The view that you use determines
    which spoke fields are included in the response.

    Values:
      SPOKE_VIEW_UNSPECIFIED: The spoke view is unspecified. When the spoke
        view is unspecified, the API returns the same fields as the `BASIC`
        view.
      BASIC: Includes `name`, `create_time`, `hub`, `unique_id`, `state`,
        `reasons`, and `spoke_type`. This is the default value.
      DETAILED: Includes all spoke fields except `labels`. You can use the
        `DETAILED` view only when you set the `spoke_locations` field to
        `[global]`.
    """
        SPOKE_VIEW_UNSPECIFIED = 0
        BASIC = 1
        DETAILED = 2
    filter = _messages.StringField(1)
    name = _messages.StringField(2, required=True)
    orderBy = _messages.StringField(3)
    pageSize = _messages.IntegerField(4, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(5)
    spokeLocations = _messages.StringField(6, repeated=True)
    view = _messages.EnumField('ViewValueValuesEnum', 7)