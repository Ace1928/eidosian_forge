from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.aiplatform.v1 import aiplatform_v1_messages as messages
class ProjectsLocationsMetadataStoresMetadataSchemasService(base_api.BaseApiService):
    """Service class for the projects_locations_metadataStores_metadataSchemas resource."""
    _NAME = 'projects_locations_metadataStores_metadataSchemas'

    def __init__(self, client):
        super(AiplatformV1.ProjectsLocationsMetadataStoresMetadataSchemasService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a MetadataSchema.

      Args:
        request: (AiplatformProjectsLocationsMetadataStoresMetadataSchemasCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1MetadataSchema) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/metadataStores/{metadataStoresId}/metadataSchemas', http_method='POST', method_id='aiplatform.projects.locations.metadataStores.metadataSchemas.create', ordered_params=['parent'], path_params=['parent'], query_params=['metadataSchemaId'], relative_path='v1/{+parent}/metadataSchemas', request_field='googleCloudAiplatformV1MetadataSchema', request_type_name='AiplatformProjectsLocationsMetadataStoresMetadataSchemasCreateRequest', response_type_name='GoogleCloudAiplatformV1MetadataSchema', supports_download=False)

    def Get(self, request, global_params=None):
        """Retrieves a specific MetadataSchema.

      Args:
        request: (AiplatformProjectsLocationsMetadataStoresMetadataSchemasGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1MetadataSchema) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/metadataStores/{metadataStoresId}/metadataSchemas/{metadataSchemasId}', http_method='GET', method_id='aiplatform.projects.locations.metadataStores.metadataSchemas.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='AiplatformProjectsLocationsMetadataStoresMetadataSchemasGetRequest', response_type_name='GoogleCloudAiplatformV1MetadataSchema', supports_download=False)

    def List(self, request, global_params=None):
        """Lists MetadataSchemas.

      Args:
        request: (AiplatformProjectsLocationsMetadataStoresMetadataSchemasListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1ListMetadataSchemasResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/metadataStores/{metadataStoresId}/metadataSchemas', http_method='GET', method_id='aiplatform.projects.locations.metadataStores.metadataSchemas.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/metadataSchemas', request_field='', request_type_name='AiplatformProjectsLocationsMetadataStoresMetadataSchemasListRequest', response_type_name='GoogleCloudAiplatformV1ListMetadataSchemasResponse', supports_download=False)