from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.configdelivery.v1alpha import configdelivery_v1alpha_messages as messages
class ProjectsLocationsResourceBundlesService(base_api.BaseApiService):
    """Service class for the projects_locations_resourceBundles resource."""
    _NAME = 'projects_locations_resourceBundles'

    def __init__(self, client):
        super(ConfigdeliveryV1alpha.ProjectsLocationsResourceBundlesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new ResourceBundle in a given project and location.

      Args:
        request: (ConfigdeliveryProjectsLocationsResourceBundlesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/resourceBundles', http_method='POST', method_id='configdelivery.projects.locations.resourceBundles.create', ordered_params=['parent'], path_params=['parent'], query_params=['requestId', 'resourceBundleId'], relative_path='v1alpha/{+parent}/resourceBundles', request_field='resourceBundle', request_type_name='ConfigdeliveryProjectsLocationsResourceBundlesCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a single ResourceBundle.

      Args:
        request: (ConfigdeliveryProjectsLocationsResourceBundlesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/resourceBundles/{resourceBundlesId}', http_method='DELETE', method_id='configdelivery.projects.locations.resourceBundles.delete', ordered_params=['name'], path_params=['name'], query_params=['force', 'requestId'], relative_path='v1alpha/{+name}', request_field='', request_type_name='ConfigdeliveryProjectsLocationsResourceBundlesDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single ResourceBundle.

      Args:
        request: (ConfigdeliveryProjectsLocationsResourceBundlesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ResourceBundle) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/resourceBundles/{resourceBundlesId}', http_method='GET', method_id='configdelivery.projects.locations.resourceBundles.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha/{+name}', request_field='', request_type_name='ConfigdeliveryProjectsLocationsResourceBundlesGetRequest', response_type_name='ResourceBundle', supports_download=False)

    def List(self, request, global_params=None):
        """Lists ResourceBundles in a given project and location.

      Args:
        request: (ConfigdeliveryProjectsLocationsResourceBundlesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListResourceBundlesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/resourceBundles', http_method='GET', method_id='configdelivery.projects.locations.resourceBundles.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1alpha/{+parent}/resourceBundles', request_field='', request_type_name='ConfigdeliveryProjectsLocationsResourceBundlesListRequest', response_type_name='ListResourceBundlesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the parameters of a single ResourceBundle.

      Args:
        request: (ConfigdeliveryProjectsLocationsResourceBundlesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/resourceBundles/{resourceBundlesId}', http_method='PATCH', method_id='configdelivery.projects.locations.resourceBundles.patch', ordered_params=['name'], path_params=['name'], query_params=['requestId', 'updateMask'], relative_path='v1alpha/{+name}', request_field='resourceBundle', request_type_name='ConfigdeliveryProjectsLocationsResourceBundlesPatchRequest', response_type_name='Operation', supports_download=False)