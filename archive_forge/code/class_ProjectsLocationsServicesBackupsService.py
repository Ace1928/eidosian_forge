from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.metastore.v1beta import metastore_v1beta_messages as messages
class ProjectsLocationsServicesBackupsService(base_api.BaseApiService):
    """Service class for the projects_locations_services_backups resource."""
    _NAME = 'projects_locations_services_backups'

    def __init__(self, client):
        super(MetastoreV1beta.ProjectsLocationsServicesBackupsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new backup in a given project and location.

      Args:
        request: (MetastoreProjectsLocationsServicesBackupsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/services/{servicesId}/backups', http_method='POST', method_id='metastore.projects.locations.services.backups.create', ordered_params=['parent'], path_params=['parent'], query_params=['backupId', 'requestId'], relative_path='v1beta/{+parent}/backups', request_field='backup', request_type_name='MetastoreProjectsLocationsServicesBackupsCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a single backup.

      Args:
        request: (MetastoreProjectsLocationsServicesBackupsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/services/{servicesId}/backups/{backupsId}', http_method='DELETE', method_id='metastore.projects.locations.services.backups.delete', ordered_params=['name'], path_params=['name'], query_params=['requestId'], relative_path='v1beta/{+name}', request_field='', request_type_name='MetastoreProjectsLocationsServicesBackupsDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single backup.

      Args:
        request: (MetastoreProjectsLocationsServicesBackupsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Backup) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/services/{servicesId}/backups/{backupsId}', http_method='GET', method_id='metastore.projects.locations.services.backups.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1beta/{+name}', request_field='', request_type_name='MetastoreProjectsLocationsServicesBackupsGetRequest', response_type_name='Backup', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the access control policy for a resource. Returns an empty policy if the resource exists and does not have a policy set.

      Args:
        request: (MetastoreProjectsLocationsServicesBackupsGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/services/{servicesId}/backups/{backupsId}:getIamPolicy', http_method='GET', method_id='metastore.projects.locations.services.backups.getIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=['options_requestedPolicyVersion'], relative_path='v1beta/{+resource}:getIamPolicy', request_field='', request_type_name='MetastoreProjectsLocationsServicesBackupsGetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def List(self, request, global_params=None):
        """Lists backups in a service.

      Args:
        request: (MetastoreProjectsLocationsServicesBackupsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListBackupsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/services/{servicesId}/backups', http_method='GET', method_id='metastore.projects.locations.services.backups.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1beta/{+parent}/backups', request_field='', request_type_name='MetastoreProjectsLocationsServicesBackupsListRequest', response_type_name='ListBackupsResponse', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the access control policy on the specified resource. Replaces any existing policy.Can return NOT_FOUND, INVALID_ARGUMENT, and PERMISSION_DENIED errors.

      Args:
        request: (MetastoreProjectsLocationsServicesBackupsSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/services/{servicesId}/backups/{backupsId}:setIamPolicy', http_method='POST', method_id='metastore.projects.locations.services.backups.setIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1beta/{+resource}:setIamPolicy', request_field='setIamPolicyRequest', request_type_name='MetastoreProjectsLocationsServicesBackupsSetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource. If the resource does not exist, this will return an empty set of permissions, not a NOT_FOUND error.Note: This operation is designed to be used for building permission-aware UIs and command-line tools, not for authorization checking. This operation may "fail open" without warning.

      Args:
        request: (MetastoreProjectsLocationsServicesBackupsTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestIamPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1beta/projects/{projectsId}/locations/{locationsId}/services/{servicesId}/backups/{backupsId}:testIamPermissions', http_method='POST', method_id='metastore.projects.locations.services.backups.testIamPermissions', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1beta/{+resource}:testIamPermissions', request_field='testIamPermissionsRequest', request_type_name='MetastoreProjectsLocationsServicesBackupsTestIamPermissionsRequest', response_type_name='TestIamPermissionsResponse', supports_download=False)