from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BeyondcorpProjectsLocationsAppConnectionsResolveRequest(_messages.Message):
    """A BeyondcorpProjectsLocationsAppConnectionsResolveRequest object.

  Fields:
    appConnectorId: Required. BeyondCorp Connector name of the connector
      associated with those AppConnections using the form: `projects/{project_
      id}/locations/{location_id}/appConnectors/{app_connector_id}`
    pageSize: Optional. The maximum number of items to return. If not
      specified, a default value of 50 will be used by the service. Regardless
      of the page_size value, the response may include a partial list and a
      caller should only rely on response's next_page_token to determine if
      there are more instances left to be queried.
    pageToken: Optional. The next_page_token value returned from a previous
      ResolveAppConnectionsResponse, if any.
    parent: Required. The resource name of the AppConnection location using
      the form: `projects/{project_id}/locations/{location_id}`
  """
    appConnectorId = _messages.StringField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    parent = _messages.StringField(4, required=True)