from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apigeeregistry.v1 import apigeeregistry_v1_messages as messages
class ProjectsLocationsApisVersionsSpecsService(base_api.BaseApiService):
    """Service class for the projects_locations_apis_versions_specs resource."""
    _NAME = 'projects_locations_apis_versions_specs'

    def __init__(self, client):
        super(ApigeeregistryV1.ProjectsLocationsApisVersionsSpecsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a specified spec.

      Args:
        request: (ApigeeregistryProjectsLocationsApisVersionsSpecsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ApiSpec) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/apis/{apisId}/versions/{versionsId}/specs', http_method='POST', method_id='apigeeregistry.projects.locations.apis.versions.specs.create', ordered_params=['parent'], path_params=['parent'], query_params=['apiSpecId'], relative_path='v1/{+parent}/specs', request_field='apiSpec', request_type_name='ApigeeregistryProjectsLocationsApisVersionsSpecsCreateRequest', response_type_name='ApiSpec', supports_download=False)

    def Delete(self, request, global_params=None):
        """Removes a specified spec, all revisions, and all child resources (e.g., artifacts).

      Args:
        request: (ApigeeregistryProjectsLocationsApisVersionsSpecsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/apis/{apisId}/versions/{versionsId}/specs/{specsId}', http_method='DELETE', method_id='apigeeregistry.projects.locations.apis.versions.specs.delete', ordered_params=['name'], path_params=['name'], query_params=['force'], relative_path='v1/{+name}', request_field='', request_type_name='ApigeeregistryProjectsLocationsApisVersionsSpecsDeleteRequest', response_type_name='Empty', supports_download=False)

    def DeleteRevision(self, request, global_params=None):
        """Deletes a revision of a spec.

      Args:
        request: (ApigeeregistryProjectsLocationsApisVersionsSpecsDeleteRevisionRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ApiSpec) The response message.
      """
        config = self.GetMethodConfig('DeleteRevision')
        return self._RunMethod(config, request, global_params=global_params)
    DeleteRevision.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/apis/{apisId}/versions/{versionsId}/specs/{specsId}:deleteRevision', http_method='DELETE', method_id='apigeeregistry.projects.locations.apis.versions.specs.deleteRevision', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:deleteRevision', request_field='', request_type_name='ApigeeregistryProjectsLocationsApisVersionsSpecsDeleteRevisionRequest', response_type_name='ApiSpec', supports_download=False)

    def Get(self, request, global_params=None):
        """Returns a specified spec.

      Args:
        request: (ApigeeregistryProjectsLocationsApisVersionsSpecsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ApiSpec) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/apis/{apisId}/versions/{versionsId}/specs/{specsId}', http_method='GET', method_id='apigeeregistry.projects.locations.apis.versions.specs.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ApigeeregistryProjectsLocationsApisVersionsSpecsGetRequest', response_type_name='ApiSpec', supports_download=False)

    def GetContents(self, request, global_params=None):
        """Returns the contents of a specified spec. If specs are stored with GZip compression, the default behavior is to return the spec uncompressed (the mime_type response field indicates the exact format returned).

      Args:
        request: (ApigeeregistryProjectsLocationsApisVersionsSpecsGetContentsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (HttpBody) The response message.
      """
        config = self.GetMethodConfig('GetContents')
        return self._RunMethod(config, request, global_params=global_params)
    GetContents.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/apis/{apisId}/versions/{versionsId}/specs/{specsId}:getContents', http_method='GET', method_id='apigeeregistry.projects.locations.apis.versions.specs.getContents', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:getContents', request_field='', request_type_name='ApigeeregistryProjectsLocationsApisVersionsSpecsGetContentsRequest', response_type_name='HttpBody', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the access control policy for a resource. Returns an empty policy if the resource exists and does not have a policy set.

      Args:
        request: (ApigeeregistryProjectsLocationsApisVersionsSpecsGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/apis/{apisId}/versions/{versionsId}/specs/{specsId}:getIamPolicy', http_method='GET', method_id='apigeeregistry.projects.locations.apis.versions.specs.getIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=['options_requestedPolicyVersion'], relative_path='v1/{+resource}:getIamPolicy', request_field='', request_type_name='ApigeeregistryProjectsLocationsApisVersionsSpecsGetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def List(self, request, global_params=None):
        """Returns matching specs.

      Args:
        request: (ApigeeregistryProjectsLocationsApisVersionsSpecsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListApiSpecsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/apis/{apisId}/versions/{versionsId}/specs', http_method='GET', method_id='apigeeregistry.projects.locations.apis.versions.specs.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/specs', request_field='', request_type_name='ApigeeregistryProjectsLocationsApisVersionsSpecsListRequest', response_type_name='ListApiSpecsResponse', supports_download=False)

    def ListRevisions(self, request, global_params=None):
        """Lists all revisions of a spec. Revisions are returned in descending order of revision creation time.

      Args:
        request: (ApigeeregistryProjectsLocationsApisVersionsSpecsListRevisionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListApiSpecRevisionsResponse) The response message.
      """
        config = self.GetMethodConfig('ListRevisions')
        return self._RunMethod(config, request, global_params=global_params)
    ListRevisions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/apis/{apisId}/versions/{versionsId}/specs/{specsId}:listRevisions', http_method='GET', method_id='apigeeregistry.projects.locations.apis.versions.specs.listRevisions', ordered_params=['name'], path_params=['name'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v1/{+name}:listRevisions', request_field='', request_type_name='ApigeeregistryProjectsLocationsApisVersionsSpecsListRevisionsRequest', response_type_name='ListApiSpecRevisionsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Used to modify a specified spec.

      Args:
        request: (ApigeeregistryProjectsLocationsApisVersionsSpecsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ApiSpec) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/apis/{apisId}/versions/{versionsId}/specs/{specsId}', http_method='PATCH', method_id='apigeeregistry.projects.locations.apis.versions.specs.patch', ordered_params=['name'], path_params=['name'], query_params=['allowMissing', 'updateMask'], relative_path='v1/{+name}', request_field='apiSpec', request_type_name='ApigeeregistryProjectsLocationsApisVersionsSpecsPatchRequest', response_type_name='ApiSpec', supports_download=False)

    def Rollback(self, request, global_params=None):
        """Sets the current revision to a specified prior revision. Note that this creates a new revision with a new revision ID.

      Args:
        request: (ApigeeregistryProjectsLocationsApisVersionsSpecsRollbackRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ApiSpec) The response message.
      """
        config = self.GetMethodConfig('Rollback')
        return self._RunMethod(config, request, global_params=global_params)
    Rollback.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/apis/{apisId}/versions/{versionsId}/specs/{specsId}:rollback', http_method='POST', method_id='apigeeregistry.projects.locations.apis.versions.specs.rollback', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:rollback', request_field='rollbackApiSpecRequest', request_type_name='ApigeeregistryProjectsLocationsApisVersionsSpecsRollbackRequest', response_type_name='ApiSpec', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the access control policy on the specified resource. Replaces any existing policy. Can return `NOT_FOUND`, `INVALID_ARGUMENT`, and `PERMISSION_DENIED` errors.

      Args:
        request: (ApigeeregistryProjectsLocationsApisVersionsSpecsSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/apis/{apisId}/versions/{versionsId}/specs/{specsId}:setIamPolicy', http_method='POST', method_id='apigeeregistry.projects.locations.apis.versions.specs.setIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:setIamPolicy', request_field='setIamPolicyRequest', request_type_name='ApigeeregistryProjectsLocationsApisVersionsSpecsSetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def TagRevision(self, request, global_params=None):
        """Adds a tag to a specified revision of a spec.

      Args:
        request: (ApigeeregistryProjectsLocationsApisVersionsSpecsTagRevisionRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ApiSpec) The response message.
      """
        config = self.GetMethodConfig('TagRevision')
        return self._RunMethod(config, request, global_params=global_params)
    TagRevision.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/apis/{apisId}/versions/{versionsId}/specs/{specsId}:tagRevision', http_method='POST', method_id='apigeeregistry.projects.locations.apis.versions.specs.tagRevision', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:tagRevision', request_field='tagApiSpecRevisionRequest', request_type_name='ApigeeregistryProjectsLocationsApisVersionsSpecsTagRevisionRequest', response_type_name='ApiSpec', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource. If the resource does not exist, this will return an empty set of permissions, not a `NOT_FOUND` error. Note: This operation is designed to be used for building permission-aware UIs and command-line tools, not for authorization checking. This operation may "fail open" without warning.

      Args:
        request: (ApigeeregistryProjectsLocationsApisVersionsSpecsTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestIamPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/apis/{apisId}/versions/{versionsId}/specs/{specsId}:testIamPermissions', http_method='POST', method_id='apigeeregistry.projects.locations.apis.versions.specs.testIamPermissions', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:testIamPermissions', request_field='testIamPermissionsRequest', request_type_name='ApigeeregistryProjectsLocationsApisVersionsSpecsTestIamPermissionsRequest', response_type_name='TestIamPermissionsResponse', supports_download=False)