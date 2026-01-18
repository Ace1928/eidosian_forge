from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.run.v1 import run_v1_messages as messages
class ProjectsLocationsSecretsService(base_api.BaseApiService):
    """Service class for the projects_locations_secrets resource."""
    _NAME = 'projects_locations_secrets'

    def __init__(self, client):
        super(RunV1.ProjectsLocationsSecretsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new secret.

      Args:
        request: (RunProjectsLocationsSecretsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Secret) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/secrets', http_method='POST', method_id='run.projects.locations.secrets.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/secrets', request_field='secret', request_type_name='RunProjectsLocationsSecretsCreateRequest', response_type_name='Secret', supports_download=False)

    def Get(self, request, global_params=None):
        """Rpc to get information about a secret.

      Args:
        request: (RunProjectsLocationsSecretsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Secret) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/secrets/{secretsId}', http_method='GET', method_id='run.projects.locations.secrets.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='RunProjectsLocationsSecretsGetRequest', response_type_name='Secret', supports_download=False)

    def ReplaceSecret(self, request, global_params=None):
        """Rpc to replace a secret. Only the spec, metadata labels, and annotations are modifiable. After the Update request, Cloud Run will work to make the 'status' match the requested 'spec'. May provide metadata.resourceVersion to enforce update from last read for optimistic concurrency control.

      Args:
        request: (RunProjectsLocationsSecretsReplaceSecretRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Secret) The response message.
      """
        config = self.GetMethodConfig('ReplaceSecret')
        return self._RunMethod(config, request, global_params=global_params)
    ReplaceSecret.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/secrets/{secretsId}', http_method='PUT', method_id='run.projects.locations.secrets.replaceSecret', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='secret', request_type_name='RunProjectsLocationsSecretsReplaceSecretRequest', response_type_name='Secret', supports_download=False)