from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.pubsublite.v1 import pubsublite_v1_messages as messages
class AdminProjectsLocationsReservationsService(base_api.BaseApiService):
    """Service class for the admin_projects_locations_reservations resource."""
    _NAME = 'admin_projects_locations_reservations'

    def __init__(self, client):
        super(PubsubliteV1.AdminProjectsLocationsReservationsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new reservation.

      Args:
        request: (PubsubliteAdminProjectsLocationsReservationsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Reservation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/admin/projects/{projectsId}/locations/{locationsId}/reservations', http_method='POST', method_id='pubsublite.admin.projects.locations.reservations.create', ordered_params=['parent'], path_params=['parent'], query_params=['reservationId'], relative_path='v1/admin/{+parent}/reservations', request_field='reservation', request_type_name='PubsubliteAdminProjectsLocationsReservationsCreateRequest', response_type_name='Reservation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes the specified reservation.

      Args:
        request: (PubsubliteAdminProjectsLocationsReservationsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/admin/projects/{projectsId}/locations/{locationsId}/reservations/{reservationsId}', http_method='DELETE', method_id='pubsublite.admin.projects.locations.reservations.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/admin/{+name}', request_field='', request_type_name='PubsubliteAdminProjectsLocationsReservationsDeleteRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns the reservation configuration.

      Args:
        request: (PubsubliteAdminProjectsLocationsReservationsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Reservation) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/admin/projects/{projectsId}/locations/{locationsId}/reservations/{reservationsId}', http_method='GET', method_id='pubsublite.admin.projects.locations.reservations.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/admin/{+name}', request_field='', request_type_name='PubsubliteAdminProjectsLocationsReservationsGetRequest', response_type_name='Reservation', supports_download=False)

    def List(self, request, global_params=None):
        """Returns the list of reservations for the given project.

      Args:
        request: (PubsubliteAdminProjectsLocationsReservationsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListReservationsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/admin/projects/{projectsId}/locations/{locationsId}/reservations', http_method='GET', method_id='pubsublite.admin.projects.locations.reservations.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1/admin/{+parent}/reservations', request_field='', request_type_name='PubsubliteAdminProjectsLocationsReservationsListRequest', response_type_name='ListReservationsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates properties of the specified reservation.

      Args:
        request: (PubsubliteAdminProjectsLocationsReservationsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Reservation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/admin/projects/{projectsId}/locations/{locationsId}/reservations/{reservationsId}', http_method='PATCH', method_id='pubsublite.admin.projects.locations.reservations.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1/admin/{+name}', request_field='reservation', request_type_name='PubsubliteAdminProjectsLocationsReservationsPatchRequest', response_type_name='Reservation', supports_download=False)