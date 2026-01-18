from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.container.v1 import container_v1_messages as messages
class ProjectsAggregatedUsableSubnetworksService(base_api.BaseApiService):
    """Service class for the projects_aggregated_usableSubnetworks resource."""
    _NAME = 'projects_aggregated_usableSubnetworks'

    def __init__(self, client):
        super(ContainerV1.ProjectsAggregatedUsableSubnetworksService, self).__init__(client)
        self._upload_configs = {}

    def List(self, request, global_params=None):
        """Lists subnetworks that are usable for creating clusters in a project.

      Args:
        request: (ContainerProjectsAggregatedUsableSubnetworksListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListUsableSubnetworksResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/aggregated/usableSubnetworks', http_method='GET', method_id='container.projects.aggregated.usableSubnetworks.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/aggregated/usableSubnetworks', request_field='', request_type_name='ContainerProjectsAggregatedUsableSubnetworksListRequest', response_type_name='ListUsableSubnetworksResponse', supports_download=False)