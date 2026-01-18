from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.gkehub.v1alpha1 import gkehub_v1alpha1_messages as messages
class ProjectsLocationsGlobalFeaturesService(base_api.BaseApiService):
    """Service class for the projects_locations_global_features resource."""
    _NAME = 'projects_locations_global_features'

    def __init__(self, client):
        super(GkehubV1alpha1.ProjectsLocationsGlobalFeaturesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Adds a new Feature.

      Args:
        request: (GkehubProjectsLocationsGlobalFeaturesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/global/features', http_method='POST', method_id='gkehub.projects.locations.global.features.create', ordered_params=['parent'], path_params=['parent'], query_params=['featureId'], relative_path='v1alpha1/{+parent}/features', request_field='feature', request_type_name='GkehubProjectsLocationsGlobalFeaturesCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Removes a Feature.

      Args:
        request: (GkehubProjectsLocationsGlobalFeaturesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/global/features/{featuresId}', http_method='DELETE', method_id='gkehub.projects.locations.global.features.delete', ordered_params=['name'], path_params=['name'], query_params=['force'], relative_path='v1alpha1/{+name}', request_field='', request_type_name='GkehubProjectsLocationsGlobalFeaturesDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single Feature.

      Args:
        request: (GkehubProjectsLocationsGlobalFeaturesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Feature) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/global/features/{featuresId}', http_method='GET', method_id='gkehub.projects.locations.global.features.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha1/{+name}', request_field='', request_type_name='GkehubProjectsLocationsGlobalFeaturesGetRequest', response_type_name='Feature', supports_download=False)

    def List(self, request, global_params=None):
        """Lists Features in a given project and location.

      Args:
        request: (GkehubProjectsLocationsGlobalFeaturesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListFeaturesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/global/features', http_method='GET', method_id='gkehub.projects.locations.global.features.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1alpha1/{+parent}/features', request_field='', request_type_name='GkehubProjectsLocationsGlobalFeaturesListRequest', response_type_name='ListFeaturesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates an existing Feature.

      Args:
        request: (GkehubProjectsLocationsGlobalFeaturesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/global/features/{featuresId}', http_method='PATCH', method_id='gkehub.projects.locations.global.features.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1alpha1/{+name}', request_field='feature', request_type_name='GkehubProjectsLocationsGlobalFeaturesPatchRequest', response_type_name='Operation', supports_download=False)