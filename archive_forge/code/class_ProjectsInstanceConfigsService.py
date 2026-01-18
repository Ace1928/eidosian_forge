from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.spanner.v1 import spanner_v1_messages as messages
class ProjectsInstanceConfigsService(base_api.BaseApiService):
    """Service class for the projects_instanceConfigs resource."""
    _NAME = 'projects_instanceConfigs'

    def __init__(self, client):
        super(SpannerV1.ProjectsInstanceConfigsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates an instance config and begins preparing it to be used. The returned long-running operation can be used to track the progress of preparing the new instance config. The instance config name is assigned by the caller. If the named instance config already exists, `CreateInstanceConfig` returns `ALREADY_EXISTS`. Immediately after the request returns: * The instance config is readable via the API, with all requested attributes. The instance config's reconciling field is set to true. Its state is `CREATING`. While the operation is pending: * Cancelling the operation renders the instance config immediately unreadable via the API. * Except for deleting the creating resource, all other attempts to modify the instance config are rejected. Upon completion of the returned operation: * Instances can be created using the instance configuration. * The instance config's reconciling field becomes false. Its state becomes `READY`. The returned long-running operation will have a name of the format `/operations/` and can be used to track creation of the instance config. The metadata field type is CreateInstanceConfigMetadata. The response field type is InstanceConfig, if successful. Authorization requires `spanner.instanceConfigs.create` permission on the resource parent.

      Args:
        request: (SpannerProjectsInstanceConfigsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/instanceConfigs', http_method='POST', method_id='spanner.projects.instanceConfigs.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/instanceConfigs', request_field='createInstanceConfigRequest', request_type_name='SpannerProjectsInstanceConfigsCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes the instance config. Deletion is only allowed when no instances are using the configuration. If any instances are using the config, returns `FAILED_PRECONDITION`. Only user managed configurations can be deleted. Authorization requires `spanner.instanceConfigs.delete` permission on the resource name.

      Args:
        request: (SpannerProjectsInstanceConfigsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/instanceConfigs/{instanceConfigsId}', http_method='DELETE', method_id='spanner.projects.instanceConfigs.delete', ordered_params=['name'], path_params=['name'], query_params=['etag', 'validateOnly'], relative_path='v1/{+name}', request_field='', request_type_name='SpannerProjectsInstanceConfigsDeleteRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets information about a particular instance configuration.

      Args:
        request: (SpannerProjectsInstanceConfigsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (InstanceConfig) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/instanceConfigs/{instanceConfigsId}', http_method='GET', method_id='spanner.projects.instanceConfigs.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='SpannerProjectsInstanceConfigsGetRequest', response_type_name='InstanceConfig', supports_download=False)

    def List(self, request, global_params=None):
        """Lists the supported instance configurations for a given project.

      Args:
        request: (SpannerProjectsInstanceConfigsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListInstanceConfigsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/instanceConfigs', http_method='GET', method_id='spanner.projects.instanceConfigs.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1/{+parent}/instanceConfigs', request_field='', request_type_name='SpannerProjectsInstanceConfigsListRequest', response_type_name='ListInstanceConfigsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates an instance config. The returned long-running operation can be used to track the progress of updating the instance. If the named instance config does not exist, returns `NOT_FOUND`. Only user managed configurations can be updated. Immediately after the request returns: * The instance config's reconciling field is set to true. While the operation is pending: * Cancelling the operation sets its metadata's cancel_time. The operation is guaranteed to succeed at undoing all changes, after which point it terminates with a `CANCELLED` status. * All other attempts to modify the instance config are rejected. * Reading the instance config via the API continues to give the pre-request values. Upon completion of the returned operation: * Creating instances using the instance configuration uses the new values. * The instance config's new values are readable via the API. * The instance config's reconciling field becomes false. The returned long-running operation will have a name of the format `/operations/` and can be used to track the instance config modification. The metadata field type is UpdateInstanceConfigMetadata. The response field type is InstanceConfig, if successful. Authorization requires `spanner.instanceConfigs.update` permission on the resource name.

      Args:
        request: (SpannerProjectsInstanceConfigsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/instanceConfigs/{instanceConfigsId}', http_method='PATCH', method_id='spanner.projects.instanceConfigs.patch', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='updateInstanceConfigRequest', request_type_name='SpannerProjectsInstanceConfigsPatchRequest', response_type_name='Operation', supports_download=False)