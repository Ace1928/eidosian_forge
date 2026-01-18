from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.spanner.v1 import spanner_v1_messages as messages
class ProjectsInstancesDatabasesService(base_api.BaseApiService):
    """Service class for the projects_instances_databases resource."""
    _NAME = 'projects_instances_databases'

    def __init__(self, client):
        super(SpannerV1.ProjectsInstancesDatabasesService, self).__init__(client)
        self._upload_configs = {}

    def Changequorum(self, request, global_params=None):
        """ChangeQuorum is strictly restricted to databases that use dual region instance configurations. Initiates a background operation to change quorum a database from dual-region mode to single-region mode and vice versa. The returned long-running operation will have a name of the format `projects//instances//databases//operations/` and can be used to track execution of the ChangeQuorum. The metadata field type is ChangeQuorumMetadata. Authorization requires `spanner.databases.changequorum` permission on the resource database.

      Args:
        request: (ChangeQuorumRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Changequorum')
        return self._RunMethod(config, request, global_params=global_params)
    Changequorum.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/instances/{instancesId}/databases/{databasesId}:changequorum', http_method='POST', method_id='spanner.projects.instances.databases.changequorum', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:changequorum', request_field='<request>', request_type_name='ChangeQuorumRequest', response_type_name='Operation', supports_download=False)

    def Create(self, request, global_params=None):
        """Creates a new Cloud Spanner database and starts to prepare it for serving. The returned long-running operation will have a name of the format `/operations/` and can be used to track preparation of the database. The metadata field type is CreateDatabaseMetadata. The response field type is Database, if successful.

      Args:
        request: (SpannerProjectsInstancesDatabasesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/instances/{instancesId}/databases', http_method='POST', method_id='spanner.projects.instances.databases.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/databases', request_field='createDatabaseRequest', request_type_name='SpannerProjectsInstancesDatabasesCreateRequest', response_type_name='Operation', supports_download=False)

    def DropDatabase(self, request, global_params=None):
        """Drops (aka deletes) a Cloud Spanner database. Completed backups for the database will be retained according to their `expire_time`. Note: Cloud Spanner might continue to accept requests for a few seconds after the database has been deleted.

      Args:
        request: (SpannerProjectsInstancesDatabasesDropDatabaseRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('DropDatabase')
        return self._RunMethod(config, request, global_params=global_params)
    DropDatabase.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/instances/{instancesId}/databases/{databasesId}', http_method='DELETE', method_id='spanner.projects.instances.databases.dropDatabase', ordered_params=['database'], path_params=['database'], query_params=[], relative_path='v1/{+database}', request_field='', request_type_name='SpannerProjectsInstancesDatabasesDropDatabaseRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets the state of a Cloud Spanner database.

      Args:
        request: (SpannerProjectsInstancesDatabasesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Database) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/instances/{instancesId}/databases/{databasesId}', http_method='GET', method_id='spanner.projects.instances.databases.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='SpannerProjectsInstancesDatabasesGetRequest', response_type_name='Database', supports_download=False)

    def GetDdl(self, request, global_params=None):
        """Returns the schema of a Cloud Spanner database as a list of formatted DDL statements. This method does not show pending schema updates, those may be queried using the Operations API.

      Args:
        request: (SpannerProjectsInstancesDatabasesGetDdlRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GetDatabaseDdlResponse) The response message.
      """
        config = self.GetMethodConfig('GetDdl')
        return self._RunMethod(config, request, global_params=global_params)
    GetDdl.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/instances/{instancesId}/databases/{databasesId}/ddl', http_method='GET', method_id='spanner.projects.instances.databases.getDdl', ordered_params=['database'], path_params=['database'], query_params=[], relative_path='v1/{+database}/ddl', request_field='', request_type_name='SpannerProjectsInstancesDatabasesGetDdlRequest', response_type_name='GetDatabaseDdlResponse', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the access control policy for a database or backup resource. Returns an empty policy if a database or backup exists but does not have a policy set. Authorization requires `spanner.databases.getIamPolicy` permission on resource. For backups, authorization requires `spanner.backups.getIamPolicy` permission on resource.

      Args:
        request: (SpannerProjectsInstancesDatabasesGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/instances/{instancesId}/databases/{databasesId}:getIamPolicy', http_method='POST', method_id='spanner.projects.instances.databases.getIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:getIamPolicy', request_field='getIamPolicyRequest', request_type_name='SpannerProjectsInstancesDatabasesGetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def GetScans(self, request, global_params=None):
        """Request a specific scan with Database-specific data for Cloud Key Visualizer.

      Args:
        request: (SpannerProjectsInstancesDatabasesGetScansRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Scan) The response message.
      """
        config = self.GetMethodConfig('GetScans')
        return self._RunMethod(config, request, global_params=global_params)
    GetScans.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/instances/{instancesId}/databases/{databasesId}/scans', http_method='GET', method_id='spanner.projects.instances.databases.getScans', ordered_params=['name'], path_params=['name'], query_params=['endTime', 'startTime', 'view'], relative_path='v1/{+name}/scans', request_field='', request_type_name='SpannerProjectsInstancesDatabasesGetScansRequest', response_type_name='Scan', supports_download=False)

    def List(self, request, global_params=None):
        """Lists Cloud Spanner databases.

      Args:
        request: (SpannerProjectsInstancesDatabasesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListDatabasesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/instances/{instancesId}/databases', http_method='GET', method_id='spanner.projects.instances.databases.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1/{+parent}/databases', request_field='', request_type_name='SpannerProjectsInstancesDatabasesListRequest', response_type_name='ListDatabasesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates a Cloud Spanner database. The returned long-running operation can be used to track the progress of updating the database. If the named database does not exist, returns `NOT_FOUND`. While the operation is pending: * The database's reconciling field is set to true. * Cancelling the operation is best-effort. If the cancellation succeeds, the operation metadata's cancel_time is set, the updates are reverted, and the operation terminates with a `CANCELLED` status. * New UpdateDatabase requests will return a `FAILED_PRECONDITION` error until the pending operation is done (returns successfully or with error). * Reading the database via the API continues to give the pre-request values. Upon completion of the returned operation: * The new values are in effect and readable via the API. * The database's reconciling field becomes false. The returned long-running operation will have a name of the format `projects//instances//databases//operations/` and can be used to track the database modification. The metadata field type is UpdateDatabaseMetadata. The response field type is Database, if successful.

      Args:
        request: (SpannerProjectsInstancesDatabasesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/instances/{instancesId}/databases/{databasesId}', http_method='PATCH', method_id='spanner.projects.instances.databases.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1/{+name}', request_field='database', request_type_name='SpannerProjectsInstancesDatabasesPatchRequest', response_type_name='Operation', supports_download=False)

    def Restore(self, request, global_params=None):
        """Create a new database by restoring from a completed backup. The new database must be in the same project and in an instance with the same instance configuration as the instance containing the backup. The returned database long-running operation has a name of the format `projects//instances//databases//operations/`, and can be used to track the progress of the operation, and to cancel it. The metadata field type is RestoreDatabaseMetadata. The response type is Database, if successful. Cancelling the returned operation will stop the restore and delete the database. There can be only one database being restored into an instance at a time. Once the restore operation completes, a new restore operation can be initiated, without waiting for the optimize operation associated with the first restore to complete.

      Args:
        request: (SpannerProjectsInstancesDatabasesRestoreRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Restore')
        return self._RunMethod(config, request, global_params=global_params)
    Restore.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/instances/{instancesId}/databases:restore', http_method='POST', method_id='spanner.projects.instances.databases.restore', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/databases:restore', request_field='restoreDatabaseRequest', request_type_name='SpannerProjectsInstancesDatabasesRestoreRequest', response_type_name='Operation', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the access control policy on a database or backup resource. Replaces any existing policy. Authorization requires `spanner.databases.setIamPolicy` permission on resource. For backups, authorization requires `spanner.backups.setIamPolicy` permission on resource.

      Args:
        request: (SpannerProjectsInstancesDatabasesSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/instances/{instancesId}/databases/{databasesId}:setIamPolicy', http_method='POST', method_id='spanner.projects.instances.databases.setIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:setIamPolicy', request_field='setIamPolicyRequest', request_type_name='SpannerProjectsInstancesDatabasesSetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that the caller has on the specified database or backup resource. Attempting this RPC on a non-existent Cloud Spanner database will result in a NOT_FOUND error if the user has `spanner.databases.list` permission on the containing Cloud Spanner instance. Otherwise returns an empty set of permissions. Calling this method on a backup that does not exist will result in a NOT_FOUND error if the user has `spanner.backups.list` permission on the containing instance.

      Args:
        request: (SpannerProjectsInstancesDatabasesTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestIamPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/instances/{instancesId}/databases/{databasesId}:testIamPermissions', http_method='POST', method_id='spanner.projects.instances.databases.testIamPermissions', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:testIamPermissions', request_field='testIamPermissionsRequest', request_type_name='SpannerProjectsInstancesDatabasesTestIamPermissionsRequest', response_type_name='TestIamPermissionsResponse', supports_download=False)

    def UpdateDdl(self, request, global_params=None):
        """Updates the schema of a Cloud Spanner database by creating/altering/dropping tables, columns, indexes, etc. The returned long-running operation will have a name of the format `/operations/` and can be used to track execution of the schema change(s). The metadata field type is UpdateDatabaseDdlMetadata. The operation has no response.

      Args:
        request: (SpannerProjectsInstancesDatabasesUpdateDdlRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('UpdateDdl')
        return self._RunMethod(config, request, global_params=global_params)
    UpdateDdl.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/instances/{instancesId}/databases/{databasesId}/ddl', http_method='PATCH', method_id='spanner.projects.instances.databases.updateDdl', ordered_params=['database'], path_params=['database'], query_params=[], relative_path='v1/{+database}/ddl', request_field='updateDatabaseDdlRequest', request_type_name='SpannerProjectsInstancesDatabasesUpdateDdlRequest', response_type_name='Operation', supports_download=False)