from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.aiplatform.v1beta1 import aiplatform_v1beta1_messages as messages
class ProjectsLocationsEndpointsChatService(base_api.BaseApiService):
    """Service class for the projects_locations_endpoints_chat resource."""
    _NAME = 'projects_locations_endpoints_chat'

    def __init__(self, client):
        super(AiplatformV1beta1.ProjectsLocationsEndpointsChatService, self).__init__(client)
        self._upload_configs = {}

    def Completions(self, request, global_params=None):
        """Exposes an OpenAI-compatible endpoint for chat completions.

      Args:
        request: (AiplatformProjectsLocationsEndpointsChatCompletionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleApiHttpBody) The response message.
      """
        config = self.GetMethodConfig('Completions')
        return self._RunMethod(config, request, global_params=global_params)
    Completions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/projects/{projectsId}/locations/{locationsId}/endpoints/{endpointsId}/chat/completions', http_method='POST', method_id='aiplatform.projects.locations.endpoints.chat.completions', ordered_params=['endpoint'], path_params=['endpoint'], query_params=[], relative_path='v1beta1/{+endpoint}/chat/completions', request_field='googleApiHttpBody', request_type_name='AiplatformProjectsLocationsEndpointsChatCompletionsRequest', response_type_name='GoogleApiHttpBody', supports_download=False)