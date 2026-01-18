from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.artifactregistry.v1 import artifactregistry_v1_messages as messages
class ProjectsLocationsRepositoriesRulesService(base_api.BaseApiService):
    """Service class for the projects_locations_repositories_rules resource."""
    _NAME = 'projects_locations_repositories_rules'

    def __init__(self, client):
        super(ArtifactregistryV1.ProjectsLocationsRepositoriesRulesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a rule.

      Args:
        request: (ArtifactregistryProjectsLocationsRepositoriesRulesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleDevtoolsArtifactregistryV1Rule) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/repositories/{repositoriesId}/rules', http_method='POST', method_id='artifactregistry.projects.locations.repositories.rules.create', ordered_params=['parent'], path_params=['parent'], query_params=['ruleId'], relative_path='v1/{+parent}/rules', request_field='googleDevtoolsArtifactregistryV1Rule', request_type_name='ArtifactregistryProjectsLocationsRepositoriesRulesCreateRequest', response_type_name='GoogleDevtoolsArtifactregistryV1Rule', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a rule.

      Args:
        request: (ArtifactregistryProjectsLocationsRepositoriesRulesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/repositories/{repositoriesId}/rules/{rulesId}', http_method='DELETE', method_id='artifactregistry.projects.locations.repositories.rules.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ArtifactregistryProjectsLocationsRepositoriesRulesDeleteRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets a rule.

      Args:
        request: (ArtifactregistryProjectsLocationsRepositoriesRulesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleDevtoolsArtifactregistryV1Rule) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/repositories/{repositoriesId}/rules/{rulesId}', http_method='GET', method_id='artifactregistry.projects.locations.repositories.rules.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ArtifactregistryProjectsLocationsRepositoriesRulesGetRequest', response_type_name='GoogleDevtoolsArtifactregistryV1Rule', supports_download=False)

    def List(self, request, global_params=None):
        """Lists rules.

      Args:
        request: (ArtifactregistryProjectsLocationsRepositoriesRulesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListRulesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/repositories/{repositoriesId}/rules', http_method='GET', method_id='artifactregistry.projects.locations.repositories.rules.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1/{+parent}/rules', request_field='', request_type_name='ArtifactregistryProjectsLocationsRepositoriesRulesListRequest', response_type_name='ListRulesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates a rule.

      Args:
        request: (ArtifactregistryProjectsLocationsRepositoriesRulesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleDevtoolsArtifactregistryV1Rule) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/repositories/{repositoriesId}/rules/{rulesId}', http_method='PATCH', method_id='artifactregistry.projects.locations.repositories.rules.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1/{+name}', request_field='googleDevtoolsArtifactregistryV1Rule', request_type_name='ArtifactregistryProjectsLocationsRepositoriesRulesPatchRequest', response_type_name='GoogleDevtoolsArtifactregistryV1Rule', supports_download=False)