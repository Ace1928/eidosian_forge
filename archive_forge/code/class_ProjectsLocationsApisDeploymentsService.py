from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apigeeregistry.v1 import apigeeregistry_v1_messages as messages
class ProjectsLocationsApisDeploymentsService(base_api.BaseApiService):
    """Service class for the projects_locations_apis_deployments resource."""
    _NAME = 'projects_locations_apis_deployments'

    def __init__(self, client):
        super(ApigeeregistryV1.ProjectsLocationsApisDeploymentsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a specified deployment.

      Args:
        request: (ApigeeregistryProjectsLocationsApisDeploymentsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ApiDeployment) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/apis/{apisId}/deployments', http_method='POST', method_id='apigeeregistry.projects.locations.apis.deployments.create', ordered_params=['parent'], path_params=['parent'], query_params=['apiDeploymentId'], relative_path='v1/{+parent}/deployments', request_field='apiDeployment', request_type_name='ApigeeregistryProjectsLocationsApisDeploymentsCreateRequest', response_type_name='ApiDeployment', supports_download=False)

    def Delete(self, request, global_params=None):
        """Removes a specified deployment, all revisions, and all child resources (e.g., artifacts).

      Args:
        request: (ApigeeregistryProjectsLocationsApisDeploymentsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/apis/{apisId}/deployments/{deploymentsId}', http_method='DELETE', method_id='apigeeregistry.projects.locations.apis.deployments.delete', ordered_params=['name'], path_params=['name'], query_params=['force'], relative_path='v1/{+name}', request_field='', request_type_name='ApigeeregistryProjectsLocationsApisDeploymentsDeleteRequest', response_type_name='Empty', supports_download=False)

    def DeleteRevision(self, request, global_params=None):
        """Deletes a revision of a deployment.

      Args:
        request: (ApigeeregistryProjectsLocationsApisDeploymentsDeleteRevisionRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ApiDeployment) The response message.
      """
        config = self.GetMethodConfig('DeleteRevision')
        return self._RunMethod(config, request, global_params=global_params)
    DeleteRevision.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/apis/{apisId}/deployments/{deploymentsId}:deleteRevision', http_method='DELETE', method_id='apigeeregistry.projects.locations.apis.deployments.deleteRevision', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:deleteRevision', request_field='', request_type_name='ApigeeregistryProjectsLocationsApisDeploymentsDeleteRevisionRequest', response_type_name='ApiDeployment', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns a specified deployment.

      Args:
        request: (ApigeeregistryProjectsLocationsApisDeploymentsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ApiDeployment) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/apis/{apisId}/deployments/{deploymentsId}', http_method='GET', method_id='apigeeregistry.projects.locations.apis.deployments.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ApigeeregistryProjectsLocationsApisDeploymentsGetRequest', response_type_name='ApiDeployment', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the access control policy for a resource. Returns an empty policy if the resource exists and does not have a policy set.

      Args:
        request: (ApigeeregistryProjectsLocationsApisDeploymentsGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/apis/{apisId}/deployments/{deploymentsId}:getIamPolicy', http_method='GET', method_id='apigeeregistry.projects.locations.apis.deployments.getIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=['options_requestedPolicyVersion'], relative_path='v1/{+resource}:getIamPolicy', request_field='', request_type_name='ApigeeregistryProjectsLocationsApisDeploymentsGetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def List(self, request, global_params=None):
        """Returns matching deployments.

      Args:
        request: (ApigeeregistryProjectsLocationsApisDeploymentsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListApiDeploymentsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/apis/{apisId}/deployments', http_method='GET', method_id='apigeeregistry.projects.locations.apis.deployments.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/deployments', request_field='', request_type_name='ApigeeregistryProjectsLocationsApisDeploymentsListRequest', response_type_name='ListApiDeploymentsResponse', supports_download=False)

    def ListRevisions(self, request, global_params=None):
        """Lists all revisions of a deployment. Revisions are returned in descending order of revision creation time.

      Args:
        request: (ApigeeregistryProjectsLocationsApisDeploymentsListRevisionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListApiDeploymentRevisionsResponse) The response message.
      """
        config = self.GetMethodConfig('ListRevisions')
        return self._RunMethod(config, request, global_params=global_params)
    ListRevisions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/apis/{apisId}/deployments/{deploymentsId}:listRevisions', http_method='GET', method_id='apigeeregistry.projects.locations.apis.deployments.listRevisions', ordered_params=['name'], path_params=['name'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v1/{+name}:listRevisions', request_field='', request_type_name='ApigeeregistryProjectsLocationsApisDeploymentsListRevisionsRequest', response_type_name='ListApiDeploymentRevisionsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Used to modify a specified deployment.

      Args:
        request: (ApigeeregistryProjectsLocationsApisDeploymentsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ApiDeployment) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/apis/{apisId}/deployments/{deploymentsId}', http_method='PATCH', method_id='apigeeregistry.projects.locations.apis.deployments.patch', ordered_params=['name'], path_params=['name'], query_params=['allowMissing', 'updateMask'], relative_path='v1/{+name}', request_field='apiDeployment', request_type_name='ApigeeregistryProjectsLocationsApisDeploymentsPatchRequest', response_type_name='ApiDeployment', supports_download=False)

    def Rollback(self, request, global_params=None):
        """Sets the current revision to a specified prior revision. Note that this creates a new revision with a new revision ID.

      Args:
        request: (ApigeeregistryProjectsLocationsApisDeploymentsRollbackRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ApiDeployment) The response message.
      """
        config = self.GetMethodConfig('Rollback')
        return self._RunMethod(config, request, global_params=global_params)
    Rollback.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/apis/{apisId}/deployments/{deploymentsId}:rollback', http_method='POST', method_id='apigeeregistry.projects.locations.apis.deployments.rollback', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:rollback', request_field='rollbackApiDeploymentRequest', request_type_name='ApigeeregistryProjectsLocationsApisDeploymentsRollbackRequest', response_type_name='ApiDeployment', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the access control policy on the specified resource. Replaces any existing policy. Can return `NOT_FOUND`, `INVALID_ARGUMENT`, and `PERMISSION_DENIED` errors.

      Args:
        request: (ApigeeregistryProjectsLocationsApisDeploymentsSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/apis/{apisId}/deployments/{deploymentsId}:setIamPolicy', http_method='POST', method_id='apigeeregistry.projects.locations.apis.deployments.setIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:setIamPolicy', request_field='setIamPolicyRequest', request_type_name='ApigeeregistryProjectsLocationsApisDeploymentsSetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def TagRevision(self, request, global_params=None):
        """Adds a tag to a specified revision of a deployment.

      Args:
        request: (ApigeeregistryProjectsLocationsApisDeploymentsTagRevisionRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ApiDeployment) The response message.
      """
        config = self.GetMethodConfig('TagRevision')
        return self._RunMethod(config, request, global_params=global_params)
    TagRevision.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/apis/{apisId}/deployments/{deploymentsId}:tagRevision', http_method='POST', method_id='apigeeregistry.projects.locations.apis.deployments.tagRevision', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:tagRevision', request_field='tagApiDeploymentRevisionRequest', request_type_name='ApigeeregistryProjectsLocationsApisDeploymentsTagRevisionRequest', response_type_name='ApiDeployment', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource. If the resource does not exist, this will return an empty set of permissions, not a `NOT_FOUND` error. Note: This operation is designed to be used for building permission-aware UIs and command-line tools, not for authorization checking. This operation may "fail open" without warning.

      Args:
        request: (ApigeeregistryProjectsLocationsApisDeploymentsTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestIamPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/apis/{apisId}/deployments/{deploymentsId}:testIamPermissions', http_method='POST', method_id='apigeeregistry.projects.locations.apis.deployments.testIamPermissions', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:testIamPermissions', request_field='testIamPermissionsRequest', request_type_name='ApigeeregistryProjectsLocationsApisDeploymentsTestIamPermissionsRequest', response_type_name='TestIamPermissionsResponse', supports_download=False)