from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.clouderrorreporting.v1beta1 import clouderrorreporting_v1beta1_messages as messages
class ProjectsGroupStatsService(base_api.BaseApiService):
    """Service class for the projects_groupStats resource."""
    _NAME = 'projects_groupStats'

    def __init__(self, client):
        super(ClouderrorreportingV1beta1.ProjectsGroupStatsService, self).__init__(client)
        self._upload_configs = {}

    def List(self, request, global_params=None):
        """Lists the specified groups.

      Args:
        request: (ClouderrorreportingProjectsGroupStatsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListGroupStatsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/projects/{projectsId}/groupStats', http_method='GET', method_id='clouderrorreporting.projects.groupStats.list', ordered_params=['projectName'], path_params=['projectName'], query_params=['alignment', 'alignmentTime', 'groupId', 'order', 'pageSize', 'pageToken', 'serviceFilter_resourceType', 'serviceFilter_service', 'serviceFilter_version', 'timeRange_period', 'timedCountDuration'], relative_path='v1beta1/{+projectName}/groupStats', request_field='', request_type_name='ClouderrorreportingProjectsGroupStatsListRequest', response_type_name='ListGroupStatsResponse', supports_download=False)