from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudbuild.v1 import cloudbuild_v1_messages as messages
class ProjectsLocationsTriggersService(base_api.BaseApiService):
    """Service class for the projects_locations_triggers resource."""
    _NAME = 'projects_locations_triggers'

    def __init__(self, client):
        super(CloudbuildV1.ProjectsLocationsTriggersService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new `BuildTrigger`.

      Args:
        request: (CloudbuildProjectsLocationsTriggersCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (BuildTrigger) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/triggers', http_method='POST', method_id='cloudbuild.projects.locations.triggers.create', ordered_params=['parent'], path_params=['parent'], query_params=['projectId'], relative_path='v1/{+parent}/triggers', request_field='buildTrigger', request_type_name='CloudbuildProjectsLocationsTriggersCreateRequest', response_type_name='BuildTrigger', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a `BuildTrigger` by its project ID and trigger ID.

      Args:
        request: (CloudbuildProjectsLocationsTriggersDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/triggers/{triggersId}', http_method='DELETE', method_id='cloudbuild.projects.locations.triggers.delete', ordered_params=['name'], path_params=['name'], query_params=['projectId', 'triggerId'], relative_path='v1/{+name}', request_field='', request_type_name='CloudbuildProjectsLocationsTriggersDeleteRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns information about a `BuildTrigger`.

      Args:
        request: (CloudbuildProjectsLocationsTriggersGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (BuildTrigger) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/triggers/{triggersId}', http_method='GET', method_id='cloudbuild.projects.locations.triggers.get', ordered_params=['name'], path_params=['name'], query_params=['projectId', 'triggerId'], relative_path='v1/{+name}', request_field='', request_type_name='CloudbuildProjectsLocationsTriggersGetRequest', response_type_name='BuildTrigger', supports_download=False)

    def List(self, request, global_params=None):
        """Lists existing `BuildTrigger`s.

      Args:
        request: (CloudbuildProjectsLocationsTriggersListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListBuildTriggersResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/triggers', http_method='GET', method_id='cloudbuild.projects.locations.triggers.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken', 'projectId'], relative_path='v1/{+parent}/triggers', request_field='', request_type_name='CloudbuildProjectsLocationsTriggersListRequest', response_type_name='ListBuildTriggersResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates a `BuildTrigger` by its project ID and trigger ID.

      Args:
        request: (CloudbuildProjectsLocationsTriggersPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (BuildTrigger) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/triggers/{triggersId}', http_method='PATCH', method_id='cloudbuild.projects.locations.triggers.patch', ordered_params=['resourceName'], path_params=['resourceName'], query_params=['projectId', 'triggerId', 'updateMask'], relative_path='v1/{+resourceName}', request_field='buildTrigger', request_type_name='CloudbuildProjectsLocationsTriggersPatchRequest', response_type_name='BuildTrigger', supports_download=False)

    def Run(self, request, global_params=None):
        """Runs a `BuildTrigger` at a particular source revision. To run a regional or global trigger, use the POST request that includes the location endpoint in the path (ex. v1/projects/{projectId}/locations/{region}/triggers/{triggerId}:run). The POST request that does not include the location endpoint in the path can only be used when running global triggers.

      Args:
        request: (CloudbuildProjectsLocationsTriggersRunRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Run')
        return self._RunMethod(config, request, global_params=global_params)
    Run.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/triggers/{triggersId}:run', http_method='POST', method_id='cloudbuild.projects.locations.triggers.run', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:run', request_field='runBuildTriggerRequest', request_type_name='CloudbuildProjectsLocationsTriggersRunRequest', response_type_name='Operation', supports_download=False)

    def Webhook(self, request, global_params=None):
        """ReceiveTriggerWebhook [Experimental] is called when the API receives a webhook request targeted at a specific trigger.

      Args:
        request: (CloudbuildProjectsLocationsTriggersWebhookRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ReceiveTriggerWebhookResponse) The response message.
      """
        config = self.GetMethodConfig('Webhook')
        return self._RunMethod(config, request, global_params=global_params)
    Webhook.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/triggers/{triggersId}:webhook', http_method='POST', method_id='cloudbuild.projects.locations.triggers.webhook', ordered_params=['name'], path_params=['name'], query_params=['projectId', 'secret', 'trigger'], relative_path='v1/{+name}:webhook', request_field='httpBody', request_type_name='CloudbuildProjectsLocationsTriggersWebhookRequest', response_type_name='ReceiveTriggerWebhookResponse', supports_download=False)