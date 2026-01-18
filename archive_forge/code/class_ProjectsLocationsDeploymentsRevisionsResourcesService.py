from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.config.v1alpha2 import config_v1alpha2_messages as messages
class ProjectsLocationsDeploymentsRevisionsResourcesService(base_api.BaseApiService):
    """Service class for the projects_locations_deployments_revisions_resources resource."""
    _NAME = 'projects_locations_deployments_revisions_resources'

    def __init__(self, client):
        super(ConfigV1alpha2.ProjectsLocationsDeploymentsRevisionsResourcesService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Gets details about a Resource deployed by Infra Manager.

      Args:
        request: (ConfigProjectsLocationsDeploymentsRevisionsResourcesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Resource) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/deployments/{deploymentsId}/revisions/{revisionsId}/resources/{resourcesId}', http_method='GET', method_id='config.projects.locations.deployments.revisions.resources.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha2/{+name}', request_field='', request_type_name='ConfigProjectsLocationsDeploymentsRevisionsResourcesGetRequest', response_type_name='Resource', supports_download=False)

    def List(self, request, global_params=None):
        """Lists Resources in a given revision.

      Args:
        request: (ConfigProjectsLocationsDeploymentsRevisionsResourcesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListResourcesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/deployments/{deploymentsId}/revisions/{revisionsId}/resources', http_method='GET', method_id='config.projects.locations.deployments.revisions.resources.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1alpha2/{+parent}/resources', request_field='', request_type_name='ConfigProjectsLocationsDeploymentsRevisionsResourcesListRequest', response_type_name='ListResourcesResponse', supports_download=False)