from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.toolresults.v1beta3 import toolresults_v1beta3_messages as messages
class ProjectsHistoriesExecutionsClustersService(base_api.BaseApiService):
    """Service class for the projects_histories_executions_clusters resource."""
    _NAME = 'projects_histories_executions_clusters'

    def __init__(self, client):
        super(ToolresultsV1beta3.ProjectsHistoriesExecutionsClustersService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Retrieves a single screenshot cluster by its ID.

      Args:
        request: (ToolresultsProjectsHistoriesExecutionsClustersGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ScreenshotCluster) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='toolresults.projects.histories.executions.clusters.get', ordered_params=['projectId', 'historyId', 'executionId', 'clusterId'], path_params=['clusterId', 'executionId', 'historyId', 'projectId'], query_params=[], relative_path='toolresults/v1beta3/projects/{projectId}/histories/{historyId}/executions/{executionId}/clusters/{clusterId}', request_field='', request_type_name='ToolresultsProjectsHistoriesExecutionsClustersGetRequest', response_type_name='ScreenshotCluster', supports_download=False)

    def List(self, request, global_params=None):
        """Lists Screenshot Clusters Returns the list of screenshot clusters corresponding to an execution. Screenshot clusters are created after the execution is finished. Clusters are created from a set of screenshots. Between any two screenshots, a matching score is calculated based off their metadata that determines how similar they are. Screenshots are placed in the cluster that has screens which have the highest matching scores.

      Args:
        request: (ToolresultsProjectsHistoriesExecutionsClustersListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListScreenshotClustersResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='toolresults.projects.histories.executions.clusters.list', ordered_params=['projectId', 'historyId', 'executionId'], path_params=['executionId', 'historyId', 'projectId'], query_params=[], relative_path='toolresults/v1beta3/projects/{projectId}/histories/{historyId}/executions/{executionId}/clusters', request_field='', request_type_name='ToolresultsProjectsHistoriesExecutionsClustersListRequest', response_type_name='ListScreenshotClustersResponse', supports_download=False)