from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.datamigration.v1 import datamigration_v1_messages as messages
class ProjectsLocationsConversionWorkspacesService(base_api.BaseApiService):
    """Service class for the projects_locations_conversionWorkspaces resource."""
    _NAME = 'projects_locations_conversionWorkspaces'

    def __init__(self, client):
        super(DatamigrationV1.ProjectsLocationsConversionWorkspacesService, self).__init__(client)
        self._upload_configs = {}

    def Apply(self, request, global_params=None):
        """Applies draft tree onto a specific destination database.

      Args:
        request: (DatamigrationProjectsLocationsConversionWorkspacesApplyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Apply')
        return self._RunMethod(config, request, global_params=global_params)
    Apply.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/conversionWorkspaces/{conversionWorkspacesId}:apply', http_method='POST', method_id='datamigration.projects.locations.conversionWorkspaces.apply', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:apply', request_field='applyConversionWorkspaceRequest', request_type_name='DatamigrationProjectsLocationsConversionWorkspacesApplyRequest', response_type_name='Operation', supports_download=False)

    def Commit(self, request, global_params=None):
        """Marks all the data in the conversion workspace as committed.

      Args:
        request: (DatamigrationProjectsLocationsConversionWorkspacesCommitRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Commit')
        return self._RunMethod(config, request, global_params=global_params)
    Commit.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/conversionWorkspaces/{conversionWorkspacesId}:commit', http_method='POST', method_id='datamigration.projects.locations.conversionWorkspaces.commit', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:commit', request_field='commitConversionWorkspaceRequest', request_type_name='DatamigrationProjectsLocationsConversionWorkspacesCommitRequest', response_type_name='Operation', supports_download=False)

    def Convert(self, request, global_params=None):
        """Creates a draft tree schema for the destination database.

      Args:
        request: (DatamigrationProjectsLocationsConversionWorkspacesConvertRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Convert')
        return self._RunMethod(config, request, global_params=global_params)
    Convert.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/conversionWorkspaces/{conversionWorkspacesId}:convert', http_method='POST', method_id='datamigration.projects.locations.conversionWorkspaces.convert', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:convert', request_field='convertConversionWorkspaceRequest', request_type_name='DatamigrationProjectsLocationsConversionWorkspacesConvertRequest', response_type_name='Operation', supports_download=False)

    def Create(self, request, global_params=None):
        """Creates a new conversion workspace in a given project and location.

      Args:
        request: (DatamigrationProjectsLocationsConversionWorkspacesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/conversionWorkspaces', http_method='POST', method_id='datamigration.projects.locations.conversionWorkspaces.create', ordered_params=['parent'], path_params=['parent'], query_params=['conversionWorkspaceId', 'requestId'], relative_path='v1/{+parent}/conversionWorkspaces', request_field='conversionWorkspace', request_type_name='DatamigrationProjectsLocationsConversionWorkspacesCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a single conversion workspace.

      Args:
        request: (DatamigrationProjectsLocationsConversionWorkspacesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/conversionWorkspaces/{conversionWorkspacesId}', http_method='DELETE', method_id='datamigration.projects.locations.conversionWorkspaces.delete', ordered_params=['name'], path_params=['name'], query_params=['force', 'requestId'], relative_path='v1/{+name}', request_field='', request_type_name='DatamigrationProjectsLocationsConversionWorkspacesDeleteRequest', response_type_name='Operation', supports_download=False)

    def DescribeConversionWorkspaceRevisions(self, request, global_params=None):
        """Retrieves a list of committed revisions of a specific conversion workspace.

      Args:
        request: (DatamigrationProjectsLocationsConversionWorkspacesDescribeConversionWorkspaceRevisionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (DescribeConversionWorkspaceRevisionsResponse) The response message.
      """
        config = self.GetMethodConfig('DescribeConversionWorkspaceRevisions')
        return self._RunMethod(config, request, global_params=global_params)
    DescribeConversionWorkspaceRevisions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/conversionWorkspaces/{conversionWorkspacesId}:describeConversionWorkspaceRevisions', http_method='GET', method_id='datamigration.projects.locations.conversionWorkspaces.describeConversionWorkspaceRevisions', ordered_params=['conversionWorkspace'], path_params=['conversionWorkspace'], query_params=['commitId'], relative_path='v1/{+conversionWorkspace}:describeConversionWorkspaceRevisions', request_field='', request_type_name='DatamigrationProjectsLocationsConversionWorkspacesDescribeConversionWorkspaceRevisionsRequest', response_type_name='DescribeConversionWorkspaceRevisionsResponse', supports_download=False)

    def DescribeDatabaseEntities(self, request, global_params=None):
        """Describes the database entities tree for a specific conversion workspace and a specific tree type. Database entities are not resources like conversion workspaces or mapping rules, and they can't be created, updated or deleted. Instead, they are simple data objects describing the structure of the client database.

      Args:
        request: (DatamigrationProjectsLocationsConversionWorkspacesDescribeDatabaseEntitiesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (DescribeDatabaseEntitiesResponse) The response message.
      """
        config = self.GetMethodConfig('DescribeDatabaseEntities')
        return self._RunMethod(config, request, global_params=global_params)
    DescribeDatabaseEntities.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/conversionWorkspaces/{conversionWorkspacesId}:describeDatabaseEntities', http_method='GET', method_id='datamigration.projects.locations.conversionWorkspaces.describeDatabaseEntities', ordered_params=['conversionWorkspace'], path_params=['conversionWorkspace'], query_params=['commitId', 'filter', 'pageSize', 'pageToken', 'tree', 'uncommitted', 'view'], relative_path='v1/{+conversionWorkspace}:describeDatabaseEntities', request_field='', request_type_name='DatamigrationProjectsLocationsConversionWorkspacesDescribeDatabaseEntitiesRequest', response_type_name='DescribeDatabaseEntitiesResponse', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single conversion workspace.

      Args:
        request: (DatamigrationProjectsLocationsConversionWorkspacesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ConversionWorkspace) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/conversionWorkspaces/{conversionWorkspacesId}', http_method='GET', method_id='datamigration.projects.locations.conversionWorkspaces.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='DatamigrationProjectsLocationsConversionWorkspacesGetRequest', response_type_name='ConversionWorkspace', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the access control policy for a resource. Returns an empty policy if the resource exists and does not have a policy set.

      Args:
        request: (DatamigrationProjectsLocationsConversionWorkspacesGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/conversionWorkspaces/{conversionWorkspacesId}:getIamPolicy', http_method='GET', method_id='datamigration.projects.locations.conversionWorkspaces.getIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=['options_requestedPolicyVersion'], relative_path='v1/{+resource}:getIamPolicy', request_field='', request_type_name='DatamigrationProjectsLocationsConversionWorkspacesGetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def List(self, request, global_params=None):
        """Lists conversion workspaces in a given project and location.

      Args:
        request: (DatamigrationProjectsLocationsConversionWorkspacesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListConversionWorkspacesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/conversionWorkspaces', http_method='GET', method_id='datamigration.projects.locations.conversionWorkspaces.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/conversionWorkspaces', request_field='', request_type_name='DatamigrationProjectsLocationsConversionWorkspacesListRequest', response_type_name='ListConversionWorkspacesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the parameters of a single conversion workspace.

      Args:
        request: (DatamigrationProjectsLocationsConversionWorkspacesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/conversionWorkspaces/{conversionWorkspacesId}', http_method='PATCH', method_id='datamigration.projects.locations.conversionWorkspaces.patch', ordered_params=['name'], path_params=['name'], query_params=['requestId', 'updateMask'], relative_path='v1/{+name}', request_field='conversionWorkspace', request_type_name='DatamigrationProjectsLocationsConversionWorkspacesPatchRequest', response_type_name='Operation', supports_download=False)

    def Rollback(self, request, global_params=None):
        """Rolls back a conversion workspace to the last committed snapshot.

      Args:
        request: (DatamigrationProjectsLocationsConversionWorkspacesRollbackRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Rollback')
        return self._RunMethod(config, request, global_params=global_params)
    Rollback.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/conversionWorkspaces/{conversionWorkspacesId}:rollback', http_method='POST', method_id='datamigration.projects.locations.conversionWorkspaces.rollback', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:rollback', request_field='rollbackConversionWorkspaceRequest', request_type_name='DatamigrationProjectsLocationsConversionWorkspacesRollbackRequest', response_type_name='Operation', supports_download=False)

    def SearchBackgroundJobs(self, request, global_params=None):
        """Searches/lists the background jobs for a specific conversion workspace. The background jobs are not resources like conversion workspaces or mapping rules, and they can't be created, updated or deleted. Instead, they are a way to expose the data plane jobs log.

      Args:
        request: (DatamigrationProjectsLocationsConversionWorkspacesSearchBackgroundJobsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SearchBackgroundJobsResponse) The response message.
      """
        config = self.GetMethodConfig('SearchBackgroundJobs')
        return self._RunMethod(config, request, global_params=global_params)
    SearchBackgroundJobs.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/conversionWorkspaces/{conversionWorkspacesId}:searchBackgroundJobs', http_method='GET', method_id='datamigration.projects.locations.conversionWorkspaces.searchBackgroundJobs', ordered_params=['conversionWorkspace'], path_params=['conversionWorkspace'], query_params=['completedUntilTime', 'maxSize', 'returnMostRecentPerJobType'], relative_path='v1/{+conversionWorkspace}:searchBackgroundJobs', request_field='', request_type_name='DatamigrationProjectsLocationsConversionWorkspacesSearchBackgroundJobsRequest', response_type_name='SearchBackgroundJobsResponse', supports_download=False)

    def Seed(self, request, global_params=None):
        """Imports a snapshot of the source database into the conversion workspace.

      Args:
        request: (DatamigrationProjectsLocationsConversionWorkspacesSeedRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Seed')
        return self._RunMethod(config, request, global_params=global_params)
    Seed.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/conversionWorkspaces/{conversionWorkspacesId}:seed', http_method='POST', method_id='datamigration.projects.locations.conversionWorkspaces.seed', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:seed', request_field='seedConversionWorkspaceRequest', request_type_name='DatamigrationProjectsLocationsConversionWorkspacesSeedRequest', response_type_name='Operation', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the access control policy on the specified resource. Replaces any existing policy. Can return `NOT_FOUND`, `INVALID_ARGUMENT`, and `PERMISSION_DENIED` errors.

      Args:
        request: (DatamigrationProjectsLocationsConversionWorkspacesSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/conversionWorkspaces/{conversionWorkspacesId}:setIamPolicy', http_method='POST', method_id='datamigration.projects.locations.conversionWorkspaces.setIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:setIamPolicy', request_field='setIamPolicyRequest', request_type_name='DatamigrationProjectsLocationsConversionWorkspacesSetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource. If the resource does not exist, this will return an empty set of permissions, not a `NOT_FOUND` error. Note: This operation is designed to be used for building permission-aware UIs and command-line tools, not for authorization checking. This operation may "fail open" without warning.

      Args:
        request: (DatamigrationProjectsLocationsConversionWorkspacesTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestIamPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/conversionWorkspaces/{conversionWorkspacesId}:testIamPermissions', http_method='POST', method_id='datamigration.projects.locations.conversionWorkspaces.testIamPermissions', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:testIamPermissions', request_field='testIamPermissionsRequest', request_type_name='DatamigrationProjectsLocationsConversionWorkspacesTestIamPermissionsRequest', response_type_name='TestIamPermissionsResponse', supports_download=False)