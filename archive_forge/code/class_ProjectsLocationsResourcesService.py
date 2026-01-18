from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.containeranalysis.v1 import containeranalysis_v1_messages as messages
class ProjectsLocationsResourcesService(base_api.BaseApiService):
    """Service class for the projects_locations_resources resource."""
    _NAME = 'projects_locations_resources'

    def __init__(self, client):
        super(ContaineranalysisV1.ProjectsLocationsResourcesService, self).__init__(client)
        self._upload_configs = {}

    def ExportSBOM(self, request, global_params=None):
        """Generates an SBOM for the given resource.

      Args:
        request: (ContaineranalysisProjectsLocationsResourcesExportSBOMRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ExportSBOMResponse) The response message.
      """
        config = self.GetMethodConfig('ExportSBOM')
        return self._RunMethod(config, request, global_params=global_params)
    ExportSBOM.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/resources/{resourcesId}:exportSBOM', http_method='POST', method_id='containeranalysis.projects.locations.resources.exportSBOM', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:exportSBOM', request_field='exportSBOMRequest', request_type_name='ContaineranalysisProjectsLocationsResourcesExportSBOMRequest', response_type_name='ExportSBOMResponse', supports_download=False)