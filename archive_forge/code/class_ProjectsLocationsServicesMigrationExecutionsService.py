from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.metastore.v1beta import metastore_v1beta_messages as messages
class ProjectsLocationsServicesMigrationExecutionsService(base_api.BaseApiService):
    """Service class for the projects_locations_services_migrationExecutions resource."""
    _NAME = 'projects_locations_services_migrationExecutions'

    def __init__(self, client):
        super(MetastoreV1beta.ProjectsLocationsServicesMigrationExecutionsService, self).__init__(client)
        self._upload_configs = {}

    def Delete(self, request, global_params=None):
        """Deletes a single migration execution.

      Args:
        request: (MetastoreProjectsLocationsServicesMigrationExecutionsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/services/{servicesId}/migrationExecutions/{migrationExecutionsId}', http_method='DELETE', method_id='metastore.projects.locations.services.migrationExecutions.delete', ordered_params=['name'], path_params=['name'], query_params=['requestId'], relative_path='v1beta/{+name}', request_field='', request_type_name='MetastoreProjectsLocationsServicesMigrationExecutionsDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single migration execution.

      Args:
        request: (MetastoreProjectsLocationsServicesMigrationExecutionsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (MigrationExecution) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/services/{servicesId}/migrationExecutions/{migrationExecutionsId}', http_method='GET', method_id='metastore.projects.locations.services.migrationExecutions.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta/{+name}', request_field='', request_type_name='MetastoreProjectsLocationsServicesMigrationExecutionsGetRequest', response_type_name='MigrationExecution', supports_download=False)

    def List(self, request, global_params=None):
        """Lists migration executions on a service.

      Args:
        request: (MetastoreProjectsLocationsServicesMigrationExecutionsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListMigrationExecutionsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/services/{servicesId}/migrationExecutions', http_method='GET', method_id='metastore.projects.locations.services.migrationExecutions.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1beta/{+parent}/migrationExecutions', request_field='', request_type_name='MetastoreProjectsLocationsServicesMigrationExecutionsListRequest', response_type_name='ListMigrationExecutionsResponse', supports_download=False)