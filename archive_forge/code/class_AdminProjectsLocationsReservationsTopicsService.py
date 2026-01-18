from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.pubsublite.v1 import pubsublite_v1_messages as messages
class AdminProjectsLocationsReservationsTopicsService(base_api.BaseApiService):
    """Service class for the admin_projects_locations_reservations_topics resource."""
    _NAME = 'admin_projects_locations_reservations_topics'

    def __init__(self, client):
        super(PubsubliteV1.AdminProjectsLocationsReservationsTopicsService, self).__init__(client)
        self._upload_configs = {}

    def List(self, request, global_params=None):
        """Lists the topics attached to the specified reservation.

      Args:
        request: (PubsubliteAdminProjectsLocationsReservationsTopicsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListReservationTopicsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/admin/projects/{projectsId}/locations/{locationsId}/reservations/{reservationsId}/topics', http_method='GET', method_id='pubsublite.admin.projects.locations.reservations.topics.list', ordered_params=['name'], path_params=['name'], query_params=['pageSize', 'pageToken'], relative_path='v1/admin/{+name}/topics', request_field='', request_type_name='PubsubliteAdminProjectsLocationsReservationsTopicsListRequest', response_type_name='ListReservationTopicsResponse', supports_download=False)