from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.spanner.v1 import spanner_v1_messages as messages
class ProjectsInstancesDatabasesDatabaseRolesService(base_api.BaseApiService):
    """Service class for the projects_instances_databases_databaseRoles resource."""
    _NAME = 'projects_instances_databases_databaseRoles'

    def __init__(self, client):
        super(SpannerV1.ProjectsInstancesDatabasesDatabaseRolesService, self).__init__(client)
        self._upload_configs = {}

    def List(self, request, global_params=None):
        """Lists Cloud Spanner database roles.

      Args:
        request: (SpannerProjectsInstancesDatabasesDatabaseRolesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListDatabaseRolesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/instances/{instancesId}/databases/{databasesId}/databaseRoles', http_method='GET', method_id='spanner.projects.instances.databases.databaseRoles.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1/{+parent}/databaseRoles', request_field='', request_type_name='SpannerProjectsInstancesDatabasesDatabaseRolesListRequest', response_type_name='ListDatabaseRolesResponse', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that the caller has on the specified database or backup resource. Attempting this RPC on a non-existent Cloud Spanner database will result in a NOT_FOUND error if the user has `spanner.databases.list` permission on the containing Cloud Spanner instance. Otherwise returns an empty set of permissions. Calling this method on a backup that does not exist will result in a NOT_FOUND error if the user has `spanner.backups.list` permission on the containing instance.

      Args:
        request: (SpannerProjectsInstancesDatabasesDatabaseRolesTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestIamPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/instances/{instancesId}/databases/{databasesId}/databaseRoles/{databaseRolesId}:testIamPermissions', http_method='POST', method_id='spanner.projects.instances.databases.databaseRoles.testIamPermissions', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:testIamPermissions', request_field='testIamPermissionsRequest', request_type_name='SpannerProjectsInstancesDatabasesDatabaseRolesTestIamPermissionsRequest', response_type_name='TestIamPermissionsResponse', supports_download=False)