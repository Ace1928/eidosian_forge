from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.composer.v1alpha2 import composer_v1alpha2_messages as messages
class ProjectsLocationsEnvironmentsUserWorkloadsSecretsService(base_api.BaseApiService):
    """Service class for the projects_locations_environments_userWorkloadsSecrets resource."""
    _NAME = 'projects_locations_environments_userWorkloadsSecrets'

    def __init__(self, client):
        super(ComposerV1alpha2.ProjectsLocationsEnvironmentsUserWorkloadsSecretsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a user workloads Secret. This method is supported for Cloud Composer environments in versions composer-3.*.*-airflow-*.*.* and newer.

      Args:
        request: (ComposerProjectsLocationsEnvironmentsUserWorkloadsSecretsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (UserWorkloadsSecret) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/environments/{environmentsId}/userWorkloadsSecrets', http_method='POST', method_id='composer.projects.locations.environments.userWorkloadsSecrets.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1alpha2/{+parent}/userWorkloadsSecrets', request_field='userWorkloadsSecret', request_type_name='ComposerProjectsLocationsEnvironmentsUserWorkloadsSecretsCreateRequest', response_type_name='UserWorkloadsSecret', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a user workloads Secret. This method is supported for Cloud Composer environments in versions composer-3.*.*-airflow-*.*.* and newer.

      Args:
        request: (ComposerProjectsLocationsEnvironmentsUserWorkloadsSecretsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/environments/{environmentsId}/userWorkloadsSecrets/{userWorkloadsSecretsId}', http_method='DELETE', method_id='composer.projects.locations.environments.userWorkloadsSecrets.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha2/{+name}', request_field='', request_type_name='ComposerProjectsLocationsEnvironmentsUserWorkloadsSecretsDeleteRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets an existing user workloads Secret. Values of the "data" field in the response are cleared. This method is supported for Cloud Composer environments in versions composer-3.*.*-airflow-*.*.* and newer.

      Args:
        request: (ComposerProjectsLocationsEnvironmentsUserWorkloadsSecretsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (UserWorkloadsSecret) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/environments/{environmentsId}/userWorkloadsSecrets/{userWorkloadsSecretsId}', http_method='GET', method_id='composer.projects.locations.environments.userWorkloadsSecrets.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha2/{+name}', request_field='', request_type_name='ComposerProjectsLocationsEnvironmentsUserWorkloadsSecretsGetRequest', response_type_name='UserWorkloadsSecret', supports_download=False)

    def List(self, request, global_params=None):
        """Lists user workloads Secrets. This method is supported for Cloud Composer environments in versions composer-3.*.*-airflow-*.*.* and newer.

      Args:
        request: (ComposerProjectsLocationsEnvironmentsUserWorkloadsSecretsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListUserWorkloadsSecretsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/environments/{environmentsId}/userWorkloadsSecrets', http_method='GET', method_id='composer.projects.locations.environments.userWorkloadsSecrets.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1alpha2/{+parent}/userWorkloadsSecrets', request_field='', request_type_name='ComposerProjectsLocationsEnvironmentsUserWorkloadsSecretsListRequest', response_type_name='ListUserWorkloadsSecretsResponse', supports_download=False)

    def Update(self, request, global_params=None):
        """Updates a user workloads Secret. This method is supported for Cloud Composer environments in versions composer-3.*.*-airflow-*.*.* and newer.

      Args:
        request: (UserWorkloadsSecret) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (UserWorkloadsSecret) The response message.
      """
        config = self.GetMethodConfig('Update')
        return self._RunMethod(config, request, global_params=global_params)
    Update.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/environments/{environmentsId}/userWorkloadsSecrets/{userWorkloadsSecretsId}', http_method='PUT', method_id='composer.projects.locations.environments.userWorkloadsSecrets.update', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha2/{+name}', request_field='<request>', request_type_name='UserWorkloadsSecret', response_type_name='UserWorkloadsSecret', supports_download=False)