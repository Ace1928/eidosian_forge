from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.composer.v1alpha2 import composer_v1alpha2_messages as messages
class ProjectsLocationsEnvironmentsUserWorkloadsConfigMapsService(base_api.BaseApiService):
    """Service class for the projects_locations_environments_userWorkloadsConfigMaps resource."""
    _NAME = 'projects_locations_environments_userWorkloadsConfigMaps'

    def __init__(self, client):
        super(ComposerV1alpha2.ProjectsLocationsEnvironmentsUserWorkloadsConfigMapsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a user workloads ConfigMap. This method is supported for Cloud Composer environments in versions composer-3.*.*-airflow-*.*.* and newer.

      Args:
        request: (ComposerProjectsLocationsEnvironmentsUserWorkloadsConfigMapsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (UserWorkloadsConfigMap) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/environments/{environmentsId}/userWorkloadsConfigMaps', http_method='POST', method_id='composer.projects.locations.environments.userWorkloadsConfigMaps.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1alpha2/{+parent}/userWorkloadsConfigMaps', request_field='userWorkloadsConfigMap', request_type_name='ComposerProjectsLocationsEnvironmentsUserWorkloadsConfigMapsCreateRequest', response_type_name='UserWorkloadsConfigMap', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a user workloads ConfigMap. This method is supported for Cloud Composer environments in versions composer-3.*.*-airflow-*.*.* and newer.

      Args:
        request: (ComposerProjectsLocationsEnvironmentsUserWorkloadsConfigMapsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/environments/{environmentsId}/userWorkloadsConfigMaps/{userWorkloadsConfigMapsId}', http_method='DELETE', method_id='composer.projects.locations.environments.userWorkloadsConfigMaps.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha2/{+name}', request_field='', request_type_name='ComposerProjectsLocationsEnvironmentsUserWorkloadsConfigMapsDeleteRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets an existing user workloads ConfigMap. This method is supported for Cloud Composer environments in versions composer-3.*.*-airflow-*.*.* and newer.

      Args:
        request: (ComposerProjectsLocationsEnvironmentsUserWorkloadsConfigMapsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (UserWorkloadsConfigMap) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/environments/{environmentsId}/userWorkloadsConfigMaps/{userWorkloadsConfigMapsId}', http_method='GET', method_id='composer.projects.locations.environments.userWorkloadsConfigMaps.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha2/{+name}', request_field='', request_type_name='ComposerProjectsLocationsEnvironmentsUserWorkloadsConfigMapsGetRequest', response_type_name='UserWorkloadsConfigMap', supports_download=False)

    def List(self, request, global_params=None):
        """Lists user workloads ConfigMaps. This method is supported for Cloud Composer environments in versions composer-3.*.*-airflow-*.*.* and newer.

      Args:
        request: (ComposerProjectsLocationsEnvironmentsUserWorkloadsConfigMapsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListUserWorkloadsConfigMapsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/environments/{environmentsId}/userWorkloadsConfigMaps', http_method='GET', method_id='composer.projects.locations.environments.userWorkloadsConfigMaps.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1alpha2/{+parent}/userWorkloadsConfigMaps', request_field='', request_type_name='ComposerProjectsLocationsEnvironmentsUserWorkloadsConfigMapsListRequest', response_type_name='ListUserWorkloadsConfigMapsResponse', supports_download=False)

    def Update(self, request, global_params=None):
        """Updates a user workloads ConfigMap. This method is supported for Cloud Composer environments in versions composer-3.*.*-airflow-*.*.* and newer.

      Args:
        request: (UserWorkloadsConfigMap) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (UserWorkloadsConfigMap) The response message.
      """
        config = self.GetMethodConfig('Update')
        return self._RunMethod(config, request, global_params=global_params)
    Update.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/environments/{environmentsId}/userWorkloadsConfigMaps/{userWorkloadsConfigMapsId}', http_method='PUT', method_id='composer.projects.locations.environments.userWorkloadsConfigMaps.update', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha2/{+name}', request_field='<request>', request_type_name='UserWorkloadsConfigMap', response_type_name='UserWorkloadsConfigMap', supports_download=False)