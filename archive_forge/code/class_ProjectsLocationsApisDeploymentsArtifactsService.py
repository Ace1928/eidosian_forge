from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apigeeregistry.v1 import apigeeregistry_v1_messages as messages
class ProjectsLocationsApisDeploymentsArtifactsService(base_api.BaseApiService):
    """Service class for the projects_locations_apis_deployments_artifacts resource."""
    _NAME = 'projects_locations_apis_deployments_artifacts'

    def __init__(self, client):
        super(ApigeeregistryV1.ProjectsLocationsApisDeploymentsArtifactsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a specified artifact.

      Args:
        request: (ApigeeregistryProjectsLocationsApisDeploymentsArtifactsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Artifact) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/apis/{apisId}/deployments/{deploymentsId}/artifacts', http_method='POST', method_id='apigeeregistry.projects.locations.apis.deployments.artifacts.create', ordered_params=['parent'], path_params=['parent'], query_params=['artifactId'], relative_path='v1/{+parent}/artifacts', request_field='artifact', request_type_name='ApigeeregistryProjectsLocationsApisDeploymentsArtifactsCreateRequest', response_type_name='Artifact', supports_download=False)

    def Delete(self, request, global_params=None):
        """Removes a specified artifact.

      Args:
        request: (ApigeeregistryProjectsLocationsApisDeploymentsArtifactsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/apis/{apisId}/deployments/{deploymentsId}/artifacts/{artifactsId}', http_method='DELETE', method_id='apigeeregistry.projects.locations.apis.deployments.artifacts.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ApigeeregistryProjectsLocationsApisDeploymentsArtifactsDeleteRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns a specified artifact.

      Args:
        request: (ApigeeregistryProjectsLocationsApisDeploymentsArtifactsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Artifact) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/apis/{apisId}/deployments/{deploymentsId}/artifacts/{artifactsId}', http_method='GET', method_id='apigeeregistry.projects.locations.apis.deployments.artifacts.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ApigeeregistryProjectsLocationsApisDeploymentsArtifactsGetRequest', response_type_name='Artifact', supports_download=False)

    def GetContents(self, request, global_params=None):
        """Returns the contents of a specified artifact. If artifacts are stored with GZip compression, the default behavior is to return the artifact uncompressed (the mime_type response field indicates the exact format returned).

      Args:
        request: (ApigeeregistryProjectsLocationsApisDeploymentsArtifactsGetContentsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (HttpBody) The response message.
      """
        config = self.GetMethodConfig('GetContents')
        return self._RunMethod(config, request, global_params=global_params)
    GetContents.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/apis/{apisId}/deployments/{deploymentsId}/artifacts/{artifactsId}:getContents', http_method='GET', method_id='apigeeregistry.projects.locations.apis.deployments.artifacts.getContents', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:getContents', request_field='', request_type_name='ApigeeregistryProjectsLocationsApisDeploymentsArtifactsGetContentsRequest', response_type_name='HttpBody', supports_download=False)

    def List(self, request, global_params=None):
        """Returns matching artifacts.

      Args:
        request: (ApigeeregistryProjectsLocationsApisDeploymentsArtifactsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListArtifactsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/apis/{apisId}/deployments/{deploymentsId}/artifacts', http_method='GET', method_id='apigeeregistry.projects.locations.apis.deployments.artifacts.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/artifacts', request_field='', request_type_name='ApigeeregistryProjectsLocationsApisDeploymentsArtifactsListRequest', response_type_name='ListArtifactsResponse', supports_download=False)

    def ReplaceArtifact(self, request, global_params=None):
        """Used to replace a specified artifact.

      Args:
        request: (Artifact) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Artifact) The response message.
      """
        config = self.GetMethodConfig('ReplaceArtifact')
        return self._RunMethod(config, request, global_params=global_params)
    ReplaceArtifact.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/apis/{apisId}/deployments/{deploymentsId}/artifacts/{artifactsId}', http_method='PUT', method_id='apigeeregistry.projects.locations.apis.deployments.artifacts.replaceArtifact', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='<request>', request_type_name='Artifact', response_type_name='Artifact', supports_download=False)