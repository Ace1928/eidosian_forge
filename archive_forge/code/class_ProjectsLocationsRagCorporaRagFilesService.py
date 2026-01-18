from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.aiplatform.v1beta1 import aiplatform_v1beta1_messages as messages
class ProjectsLocationsRagCorporaRagFilesService(base_api.BaseApiService):
    """Service class for the projects_locations_ragCorpora_ragFiles resource."""
    _NAME = 'projects_locations_ragCorpora_ragFiles'

    def __init__(self, client):
        super(AiplatformV1beta1.ProjectsLocationsRagCorporaRagFilesService, self).__init__(client)
        self._upload_configs = {}

    def Delete(self, request, global_params=None):
        """Deletes a RagFile.

      Args:
        request: (AiplatformProjectsLocationsRagCorporaRagFilesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/projects/{projectsId}/locations/{locationsId}/ragCorpora/{ragCorporaId}/ragFiles/{ragFilesId}', http_method='DELETE', method_id='aiplatform.projects.locations.ragCorpora.ragFiles.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta1/{+name}', request_field='', request_type_name='AiplatformProjectsLocationsRagCorporaRagFilesDeleteRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets a RagFile.

      Args:
        request: (AiplatformProjectsLocationsRagCorporaRagFilesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1beta1RagFile) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/projects/{projectsId}/locations/{locationsId}/ragCorpora/{ragCorporaId}/ragFiles/{ragFilesId}', http_method='GET', method_id='aiplatform.projects.locations.ragCorpora.ragFiles.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta1/{+name}', request_field='', request_type_name='AiplatformProjectsLocationsRagCorporaRagFilesGetRequest', response_type_name='GoogleCloudAiplatformV1beta1RagFile', supports_download=False)

    def Import(self, request, global_params=None):
        """Import files from Google Cloud Storage or Google Drive into a RagCorpus.

      Args:
        request: (AiplatformProjectsLocationsRagCorporaRagFilesImportRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Import')
        return self._RunMethod(config, request, global_params=global_params)
    Import.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/projects/{projectsId}/locations/{locationsId}/ragCorpora/{ragCorporaId}/ragFiles:import', http_method='POST', method_id='aiplatform.projects.locations.ragCorpora.ragFiles.import', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1beta1/{+parent}/ragFiles:import', request_field='googleCloudAiplatformV1beta1ImportRagFilesRequest', request_type_name='AiplatformProjectsLocationsRagCorporaRagFilesImportRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def List(self, request, global_params=None):
        """Lists RagFiles in a RagCorpus.

      Args:
        request: (AiplatformProjectsLocationsRagCorporaRagFilesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1beta1ListRagFilesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta1/projects/{projectsId}/locations/{locationsId}/ragCorpora/{ragCorporaId}/ragFiles', http_method='GET', method_id='aiplatform.projects.locations.ragCorpora.ragFiles.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1beta1/{+parent}/ragFiles', request_field='', request_type_name='AiplatformProjectsLocationsRagCorporaRagFilesListRequest', response_type_name='GoogleCloudAiplatformV1beta1ListRagFilesResponse', supports_download=False)