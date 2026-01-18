from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.healthcare.v1alpha2 import healthcare_v1alpha2_messages as messages
class ProjectsLocationsServicesNlpService(base_api.BaseApiService):
    """Service class for the projects_locations_services_nlp resource."""
    _NAME = 'projects_locations_services_nlp'

    def __init__(self, client):
        super(HealthcareV1alpha2.ProjectsLocationsServicesNlpService, self).__init__(client)
        self._upload_configs = {}

    def AnalyzeEntities(self, request, global_params=None):
        """Analyze heathcare entity in a document. Its response includes the recognized entity mentions and the relationships between them. AnalyzeEntities uses context aware models to detect entities.

      Args:
        request: (HealthcareProjectsLocationsServicesNlpAnalyzeEntitiesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (AnalyzeEntitiesResponse) The response message.
      """
        config = self.GetMethodConfig('AnalyzeEntities')
        return self._RunMethod(config, request, global_params=global_params)
    AnalyzeEntities.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/services/nlp:analyzeEntities', http_method='POST', method_id='healthcare.projects.locations.services.nlp.analyzeEntities', ordered_params=['nlpService'], path_params=['nlpService'], query_params=[], relative_path='v1alpha2/{+nlpService}:analyzeEntities', request_field='analyzeEntitiesRequest', request_type_name='HealthcareProjectsLocationsServicesNlpAnalyzeEntitiesRequest', response_type_name='AnalyzeEntitiesResponse', supports_download=False)