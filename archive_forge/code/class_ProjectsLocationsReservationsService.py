from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.tpu.v2alpha1 import tpu_v2alpha1_messages as messages
class ProjectsLocationsReservationsService(base_api.BaseApiService):
    """Service class for the projects_locations_reservations resource."""
    _NAME = 'projects_locations_reservations'

    def __init__(self, client):
        super(TpuV2alpha1.ProjectsLocationsReservationsService, self).__init__(client)
        self._upload_configs = {}

    def List(self, request, global_params=None):
        """Retrieves the reservations for the given project in the given location.

      Args:
        request: (TpuProjectsLocationsReservationsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListReservationsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2alpha1/projects/{projectsId}/locations/{locationsId}/reservations', http_method='GET', method_id='tpu.projects.locations.reservations.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v2alpha1/{+parent}/reservations', request_field='', request_type_name='TpuProjectsLocationsReservationsListRequest', response_type_name='ListReservationsResponse', supports_download=False)