from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.spanner.v1 import spanner_v1_messages as messages
class ProjectsInstancesInstancePartitionsService(base_api.BaseApiService):
    """Service class for the projects_instances_instancePartitions resource."""
    _NAME = 'projects_instances_instancePartitions'

    def __init__(self, client):
        super(SpannerV1.ProjectsInstancesInstancePartitionsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates an instance partition and begins preparing it to be used. The returned long-running operation can be used to track the progress of preparing the new instance partition. The instance partition name is assigned by the caller. If the named instance partition already exists, `CreateInstancePartition` returns `ALREADY_EXISTS`. Immediately upon completion of this request: * The instance partition is readable via the API, with all requested attributes but no allocated resources. Its state is `CREATING`. Until completion of the returned operation: * Cancelling the operation renders the instance partition immediately unreadable via the API. * The instance partition can be deleted. * All other attempts to modify the instance partition are rejected. Upon completion of the returned operation: * Billing for all successfully-allocated resources begins (some types may have lower than the requested levels). * Databases can start using this instance partition. * The instance partition's allocated resource levels are readable via the API. * The instance partition's state becomes `READY`. The returned long-running operation will have a name of the format `/operations/` and can be used to track creation of the instance partition. The metadata field type is CreateInstancePartitionMetadata. The response field type is InstancePartition, if successful.

      Args:
        request: (SpannerProjectsInstancesInstancePartitionsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/instances/{instancesId}/instancePartitions', http_method='POST', method_id='spanner.projects.instances.instancePartitions.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/instancePartitions', request_field='createInstancePartitionRequest', request_type_name='SpannerProjectsInstancesInstancePartitionsCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes an existing instance partition. Requires that the instance partition is not used by any database or backup and is not the default instance partition of an instance. Authorization requires `spanner.instancePartitions.delete` permission on the resource name.

      Args:
        request: (SpannerProjectsInstancesInstancePartitionsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/instances/{instancesId}/instancePartitions/{instancePartitionsId}', http_method='DELETE', method_id='spanner.projects.instances.instancePartitions.delete', ordered_params=['name'], path_params=['name'], query_params=['etag'], relative_path='v1/{+name}', request_field='', request_type_name='SpannerProjectsInstancesInstancePartitionsDeleteRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets information about a particular instance partition.

      Args:
        request: (SpannerProjectsInstancesInstancePartitionsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (InstancePartition) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/instances/{instancesId}/instancePartitions/{instancePartitionsId}', http_method='GET', method_id='spanner.projects.instances.instancePartitions.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='SpannerProjectsInstancesInstancePartitionsGetRequest', response_type_name='InstancePartition', supports_download=False)

    def List(self, request, global_params=None):
        """Lists all instance partitions for the given instance.

      Args:
        request: (SpannerProjectsInstancesInstancePartitionsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListInstancePartitionsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/instances/{instancesId}/instancePartitions', http_method='GET', method_id='spanner.projects.instances.instancePartitions.list', ordered_params=['parent'], path_params=['parent'], query_params=['instancePartitionDeadline', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/instancePartitions', request_field='', request_type_name='SpannerProjectsInstancesInstancePartitionsListRequest', response_type_name='ListInstancePartitionsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates an instance partition, and begins allocating or releasing resources as requested. The returned long-running operation can be used to track the progress of updating the instance partition. If the named instance partition does not exist, returns `NOT_FOUND`. Immediately upon completion of this request: * For resource types for which a decrease in the instance partition's allocation has been requested, billing is based on the newly-requested level. Until completion of the returned operation: * Cancelling the operation sets its metadata's cancel_time, and begins restoring resources to their pre-request values. The operation is guaranteed to succeed at undoing all resource changes, after which point it terminates with a `CANCELLED` status. * All other attempts to modify the instance partition are rejected. * Reading the instance partition via the API continues to give the pre-request resource levels. Upon completion of the returned operation: * Billing begins for all successfully-allocated resources (some types may have lower than the requested levels). * All newly-reserved resources are available for serving the instance partition's tables. * The instance partition's new resource levels are readable via the API. The returned long-running operation will have a name of the format `/operations/` and can be used to track the instance partition modification. The metadata field type is UpdateInstancePartitionMetadata. The response field type is InstancePartition, if successful. Authorization requires `spanner.instancePartitions.update` permission on the resource name.

      Args:
        request: (SpannerProjectsInstancesInstancePartitionsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/instances/{instancesId}/instancePartitions/{instancePartitionsId}', http_method='PATCH', method_id='spanner.projects.instances.instancePartitions.patch', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='updateInstancePartitionRequest', request_type_name='SpannerProjectsInstancesInstancePartitionsPatchRequest', response_type_name='Operation', supports_download=False)