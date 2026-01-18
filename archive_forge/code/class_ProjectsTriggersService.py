from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudbuild.v1 import cloudbuild_v1_messages as messages
class ProjectsTriggersService(base_api.BaseApiService):
    """Service class for the projects_triggers resource."""
    _NAME = 'projects_triggers'

    def __init__(self, client):
        super(CloudbuildV1.ProjectsTriggersService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new `BuildTrigger`.

      Args:
        request: (CloudbuildProjectsTriggersCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (BuildTrigger) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='cloudbuild.projects.triggers.create', ordered_params=['projectId'], path_params=['projectId'], query_params=['parent'], relative_path='v1/projects/{projectId}/triggers', request_field='buildTrigger', request_type_name='CloudbuildProjectsTriggersCreateRequest', response_type_name='BuildTrigger', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a `BuildTrigger` by its project ID and trigger ID.

      Args:
        request: (CloudbuildProjectsTriggersDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(http_method='DELETE', method_id='cloudbuild.projects.triggers.delete', ordered_params=['projectId', 'triggerId'], path_params=['projectId', 'triggerId'], query_params=['name'], relative_path='v1/projects/{projectId}/triggers/{triggerId}', request_field='', request_type_name='CloudbuildProjectsTriggersDeleteRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns information about a `BuildTrigger`.

      Args:
        request: (CloudbuildProjectsTriggersGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (BuildTrigger) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='cloudbuild.projects.triggers.get', ordered_params=['projectId', 'triggerId'], path_params=['projectId', 'triggerId'], query_params=['name'], relative_path='v1/projects/{projectId}/triggers/{triggerId}', request_field='', request_type_name='CloudbuildProjectsTriggersGetRequest', response_type_name='BuildTrigger', supports_download=False)

    def List(self, request, global_params=None):
        """Lists existing `BuildTrigger`s.

      Args:
        request: (CloudbuildProjectsTriggersListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListBuildTriggersResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='cloudbuild.projects.triggers.list', ordered_params=['projectId'], path_params=['projectId'], query_params=['pageSize', 'pageToken', 'parent'], relative_path='v1/projects/{projectId}/triggers', request_field='', request_type_name='CloudbuildProjectsTriggersListRequest', response_type_name='ListBuildTriggersResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates a `BuildTrigger` by its project ID and trigger ID.

      Args:
        request: (CloudbuildProjectsTriggersPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (BuildTrigger) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(http_method='PATCH', method_id='cloudbuild.projects.triggers.patch', ordered_params=['projectId', 'triggerId'], path_params=['projectId', 'triggerId'], query_params=['updateMask'], relative_path='v1/projects/{projectId}/triggers/{triggerId}', request_field='buildTrigger', request_type_name='CloudbuildProjectsTriggersPatchRequest', response_type_name='BuildTrigger', supports_download=False)

    def Run(self, request, global_params=None):
        """Runs a `BuildTrigger` at a particular source revision. To run a regional or global trigger, use the POST request that includes the location endpoint in the path (ex. v1/projects/{projectId}/locations/{region}/triggers/{triggerId}:run). The POST request that does not include the location endpoint in the path can only be used when running global triggers.

      Args:
        request: (CloudbuildProjectsTriggersRunRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Run')
        return self._RunMethod(config, request, global_params=global_params)
    Run.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='cloudbuild.projects.triggers.run', ordered_params=['projectId', 'triggerId'], path_params=['projectId', 'triggerId'], query_params=['name'], relative_path='v1/projects/{projectId}/triggers/{triggerId}:run', request_field='repoSource', request_type_name='CloudbuildProjectsTriggersRunRequest', response_type_name='Operation', supports_download=False)

    def Webhook(self, request, global_params=None):
        """ReceiveTriggerWebhook [Experimental] is called when the API receives a webhook request targeted at a specific trigger.

      Args:
        request: (CloudbuildProjectsTriggersWebhookRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ReceiveTriggerWebhookResponse) The response message.
      """
        config = self.GetMethodConfig('Webhook')
        return self._RunMethod(config, request, global_params=global_params)
    Webhook.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='cloudbuild.projects.triggers.webhook', ordered_params=['projectId', 'trigger'], path_params=['projectId', 'trigger'], query_params=['name', 'secret'], relative_path='v1/projects/{projectId}/triggers/{trigger}:webhook', request_field='httpBody', request_type_name='CloudbuildProjectsTriggersWebhookRequest', response_type_name='ReceiveTriggerWebhookResponse', supports_download=False)