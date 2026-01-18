from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.config.v1alpha2 import config_v1alpha2_messages as messages
class ProjectsLocationsDeploymentsService(base_api.BaseApiService):
    """Service class for the projects_locations_deployments resource."""
    _NAME = 'projects_locations_deployments'

    def __init__(self, client):
        super(ConfigV1alpha2.ProjectsLocationsDeploymentsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a Deployment.

      Args:
        request: (ConfigProjectsLocationsDeploymentsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/deployments', http_method='POST', method_id='config.projects.locations.deployments.create', ordered_params=['parent'], path_params=['parent'], query_params=['deploymentId', 'requestId'], relative_path='v1alpha2/{+parent}/deployments', request_field='deployment', request_type_name='ConfigProjectsLocationsDeploymentsCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a Deployment.

      Args:
        request: (ConfigProjectsLocationsDeploymentsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/deployments/{deploymentsId}', http_method='DELETE', method_id='config.projects.locations.deployments.delete', ordered_params=['name'], path_params=['name'], query_params=['deletePolicy', 'force', 'requestId'], relative_path='v1alpha2/{+name}', request_field='', request_type_name='ConfigProjectsLocationsDeploymentsDeleteRequest', response_type_name='Operation', supports_download=False)

    def DeleteState(self, request, global_params=None):
        """Deletes Terraform state file in a given deployment.

      Args:
        request: (ConfigProjectsLocationsDeploymentsDeleteStateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('DeleteState')
        return self._RunMethod(config, request, global_params=global_params)
    DeleteState.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/deployments/{deploymentsId}:deleteState', http_method='POST', method_id='config.projects.locations.deployments.deleteState', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha2/{+name}:deleteState', request_field='deleteStatefileRequest', request_type_name='ConfigProjectsLocationsDeploymentsDeleteStateRequest', response_type_name='Empty', supports_download=False)

    def ExportLock(self, request, global_params=None):
        """Exports the lock info on a locked deployment.

      Args:
        request: (ConfigProjectsLocationsDeploymentsExportLockRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (LockInfo) The response message.
      """
        config = self.GetMethodConfig('ExportLock')
        return self._RunMethod(config, request, global_params=global_params)
    ExportLock.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/deployments/{deploymentsId}:exportLock', http_method='GET', method_id='config.projects.locations.deployments.exportLock', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha2/{+name}:exportLock', request_field='', request_type_name='ConfigProjectsLocationsDeploymentsExportLockRequest', response_type_name='LockInfo', supports_download=False)

    def ExportState(self, request, global_params=None):
        """Exports Terraform state file from a given deployment.

      Args:
        request: (ConfigProjectsLocationsDeploymentsExportStateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Statefile) The response message.
      """
        config = self.GetMethodConfig('ExportState')
        return self._RunMethod(config, request, global_params=global_params)
    ExportState.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/deployments/{deploymentsId}:exportState', http_method='POST', method_id='config.projects.locations.deployments.exportState', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1alpha2/{+parent}:exportState', request_field='exportDeploymentStatefileRequest', request_type_name='ConfigProjectsLocationsDeploymentsExportStateRequest', response_type_name='Statefile', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details about a Deployment.

      Args:
        request: (ConfigProjectsLocationsDeploymentsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Deployment) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/deployments/{deploymentsId}', http_method='GET', method_id='config.projects.locations.deployments.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha2/{+name}', request_field='', request_type_name='ConfigProjectsLocationsDeploymentsGetRequest', response_type_name='Deployment', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the access control policy for a resource. Returns an empty policy if the resource exists and does not have a policy set.

      Args:
        request: (ConfigProjectsLocationsDeploymentsGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/deployments/{deploymentsId}:getIamPolicy', http_method='GET', method_id='config.projects.locations.deployments.getIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=['options_requestedPolicyVersion'], relative_path='v1alpha2/{+resource}:getIamPolicy', request_field='', request_type_name='ConfigProjectsLocationsDeploymentsGetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def ImportState(self, request, global_params=None):
        """Imports Terraform state file in a given deployment. The state file does not take effect until the Deployment has been unlocked.

      Args:
        request: (ConfigProjectsLocationsDeploymentsImportStateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Statefile) The response message.
      """
        config = self.GetMethodConfig('ImportState')
        return self._RunMethod(config, request, global_params=global_params)
    ImportState.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/deployments/{deploymentsId}:importState', http_method='POST', method_id='config.projects.locations.deployments.importState', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1alpha2/{+parent}:importState', request_field='importStatefileRequest', request_type_name='ConfigProjectsLocationsDeploymentsImportStateRequest', response_type_name='Statefile', supports_download=False)

    def List(self, request, global_params=None):
        """Lists Deployments in a given project and location.

      Args:
        request: (ConfigProjectsLocationsDeploymentsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListDeploymentsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/deployments', http_method='GET', method_id='config.projects.locations.deployments.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1alpha2/{+parent}/deployments', request_field='', request_type_name='ConfigProjectsLocationsDeploymentsListRequest', response_type_name='ListDeploymentsResponse', supports_download=False)

    def Lock(self, request, global_params=None):
        """Locks a deployment.

      Args:
        request: (ConfigProjectsLocationsDeploymentsLockRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Lock')
        return self._RunMethod(config, request, global_params=global_params)
    Lock.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/deployments/{deploymentsId}:lock', http_method='POST', method_id='config.projects.locations.deployments.lock', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha2/{+name}:lock', request_field='lockDeploymentRequest', request_type_name='ConfigProjectsLocationsDeploymentsLockRequest', response_type_name='Operation', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates a Deployment.

      Args:
        request: (ConfigProjectsLocationsDeploymentsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/deployments/{deploymentsId}', http_method='PATCH', method_id='config.projects.locations.deployments.patch', ordered_params=['name'], path_params=['name'], query_params=['requestId', 'updateMask'], relative_path='v1alpha2/{+name}', request_field='deployment', request_type_name='ConfigProjectsLocationsDeploymentsPatchRequest', response_type_name='Operation', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the access control policy on the specified resource. Replaces any existing policy. Can return `NOT_FOUND`, `INVALID_ARGUMENT`, and `PERMISSION_DENIED` errors.

      Args:
        request: (ConfigProjectsLocationsDeploymentsSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/deployments/{deploymentsId}:setIamPolicy', http_method='POST', method_id='config.projects.locations.deployments.setIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1alpha2/{+resource}:setIamPolicy', request_field='setIamPolicyRequest', request_type_name='ConfigProjectsLocationsDeploymentsSetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource. If the resource does not exist, this will return an empty set of permissions, not a `NOT_FOUND` error. Note: This operation is designed to be used for building permission-aware UIs and command-line tools, not for authorization checking. This operation may "fail open" without warning.

      Args:
        request: (ConfigProjectsLocationsDeploymentsTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestIamPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/deployments/{deploymentsId}:testIamPermissions', http_method='POST', method_id='config.projects.locations.deployments.testIamPermissions', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1alpha2/{+resource}:testIamPermissions', request_field='testIamPermissionsRequest', request_type_name='ConfigProjectsLocationsDeploymentsTestIamPermissionsRequest', response_type_name='TestIamPermissionsResponse', supports_download=False)

    def Unlock(self, request, global_params=None):
        """Unlocks a locked deployment.

      Args:
        request: (ConfigProjectsLocationsDeploymentsUnlockRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Unlock')
        return self._RunMethod(config, request, global_params=global_params)
    Unlock.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha2/projects/{projectsId}/locations/{locationsId}/deployments/{deploymentsId}:unlock', http_method='POST', method_id='config.projects.locations.deployments.unlock', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha2/{+name}:unlock', request_field='unlockDeploymentRequest', request_type_name='ConfigProjectsLocationsDeploymentsUnlockRequest', response_type_name='Operation', supports_download=False)