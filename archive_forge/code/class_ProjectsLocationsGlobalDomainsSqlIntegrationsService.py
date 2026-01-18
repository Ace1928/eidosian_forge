from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.managedidentities.v1 import managedidentities_v1_messages as messages
class ProjectsLocationsGlobalDomainsSqlIntegrationsService(base_api.BaseApiService):
    """Service class for the projects_locations_global_domains_sqlIntegrations resource."""
    _NAME = 'projects_locations_global_domains_sqlIntegrations'

    def __init__(self, client):
        super(ManagedidentitiesV1.ProjectsLocationsGlobalDomainsSqlIntegrationsService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Gets details of a single sqlIntegration.

      Args:
        request: (ManagedidentitiesProjectsLocationsGlobalDomainsSqlIntegrationsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SqlIntegration) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/global/domains/{domainsId}/sqlIntegrations/{sqlIntegrationsId}', http_method='GET', method_id='managedidentities.projects.locations.global.domains.sqlIntegrations.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ManagedidentitiesProjectsLocationsGlobalDomainsSqlIntegrationsGetRequest', response_type_name='SqlIntegration', supports_download=False)

    def List(self, request, global_params=None):
        """Lists SqlIntegrations in a given domain.

      Args:
        request: (ManagedidentitiesProjectsLocationsGlobalDomainsSqlIntegrationsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListSqlIntegrationsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/global/domains/{domainsId}/sqlIntegrations', http_method='GET', method_id='managedidentities.projects.locations.global.domains.sqlIntegrations.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/sqlIntegrations', request_field='', request_type_name='ManagedidentitiesProjectsLocationsGlobalDomainsSqlIntegrationsListRequest', response_type_name='ListSqlIntegrationsResponse', supports_download=False)