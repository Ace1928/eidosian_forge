from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.spanner.v1 import spanner_v1_messages as messages
class ProjectsInstancesInstancePartitionOperationsService(base_api.BaseApiService):
    """Service class for the projects_instances_instancePartitionOperations resource."""
    _NAME = 'projects_instances_instancePartitionOperations'

    def __init__(self, client):
        super(SpannerV1.ProjectsInstancesInstancePartitionOperationsService, self).__init__(client)
        self._upload_configs = {}

    def List(self, request, global_params=None):
        """Lists instance partition long-running operations in the given instance. An instance partition operation has a name of the form `projects//instances//instancePartitions//operations/`. The long-running operation metadata field type `metadata.type_url` describes the type of the metadata. Operations returned include those that have completed/failed/canceled within the last 7 days, and pending operations. Operations returned are ordered by `operation.metadata.value.start_time` in descending order starting from the most recently started operation. Authorization requires `spanner.instancePartitionOperations.list` permission on the resource parent.

      Args:
        request: (SpannerProjectsInstancesInstancePartitionOperationsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListInstancePartitionOperationsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/instances/{instancesId}/instancePartitionOperations', http_method='GET', method_id='spanner.projects.instances.instancePartitionOperations.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'instancePartitionDeadline', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/instancePartitionOperations', request_field='', request_type_name='SpannerProjectsInstancesInstancePartitionOperationsListRequest', response_type_name='ListInstancePartitionOperationsResponse', supports_download=False)