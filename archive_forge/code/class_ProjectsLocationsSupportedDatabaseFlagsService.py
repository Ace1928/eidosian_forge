from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.alloydb.v1beta import alloydb_v1beta_messages as messages
class ProjectsLocationsSupportedDatabaseFlagsService(base_api.BaseApiService):
    """Service class for the projects_locations_supportedDatabaseFlags resource."""
    _NAME = 'projects_locations_supportedDatabaseFlags'

    def __init__(self, client):
        super(AlloydbV1beta.ProjectsLocationsSupportedDatabaseFlagsService, self).__init__(client)
        self._upload_configs = {}

    def List(self, request, global_params=None):
        """Lists SupportedDatabaseFlags for a given project and location.

      Args:
        request: (AlloydbProjectsLocationsSupportedDatabaseFlagsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListSupportedDatabaseFlagsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/supportedDatabaseFlags', http_method='GET', method_id='alloydb.projects.locations.supportedDatabaseFlags.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1beta/{+parent}/supportedDatabaseFlags', request_field='', request_type_name='AlloydbProjectsLocationsSupportedDatabaseFlagsListRequest', response_type_name='ListSupportedDatabaseFlagsResponse', supports_download=False)