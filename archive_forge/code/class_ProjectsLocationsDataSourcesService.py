from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.bigquerydatatransfer.v1 import bigquerydatatransfer_v1_messages as messages
class ProjectsLocationsDataSourcesService(base_api.BaseApiService):
    """Service class for the projects_locations_dataSources resource."""
    _NAME = 'projects_locations_dataSources'

    def __init__(self, client):
        super(BigquerydatatransferV1.ProjectsLocationsDataSourcesService, self).__init__(client)
        self._upload_configs = {}

    def CheckValidCreds(self, request, global_params=None):
        """Returns true if valid credentials exist for the given data source and requesting user.

      Args:
        request: (BigquerydatatransferProjectsLocationsDataSourcesCheckValidCredsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (CheckValidCredsResponse) The response message.
      """
        config = self.GetMethodConfig('CheckValidCreds')
        return self._RunMethod(config, request, global_params=global_params)
    CheckValidCreds.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/dataSources/{dataSourcesId}:checkValidCreds', http_method='POST', method_id='bigquerydatatransfer.projects.locations.dataSources.checkValidCreds', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:checkValidCreds', request_field='checkValidCredsRequest', request_type_name='BigquerydatatransferProjectsLocationsDataSourcesCheckValidCredsRequest', response_type_name='CheckValidCredsResponse', supports_download=False)

    def Get(self, request, global_params=None):
        """Retrieves a supported data source and returns its settings.

      Args:
        request: (BigquerydatatransferProjectsLocationsDataSourcesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (DataSource) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/dataSources/{dataSourcesId}', http_method='GET', method_id='bigquerydatatransfer.projects.locations.dataSources.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='BigquerydatatransferProjectsLocationsDataSourcesGetRequest', response_type_name='DataSource', supports_download=False)

    def List(self, request, global_params=None):
        """Lists supported data sources and returns their settings.

      Args:
        request: (BigquerydatatransferProjectsLocationsDataSourcesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListDataSourcesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/dataSources', http_method='GET', method_id='bigquerydatatransfer.projects.locations.dataSources.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1/{+parent}/dataSources', request_field='', request_type_name='BigquerydatatransferProjectsLocationsDataSourcesListRequest', response_type_name='ListDataSourcesResponse', supports_download=False)