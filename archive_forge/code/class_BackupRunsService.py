from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.sqladmin.v1beta4 import sqladmin_v1beta4_messages as messages
class BackupRunsService(base_api.BaseApiService):
    """Service class for the backupRuns resource."""
    _NAME = 'backupRuns'

    def __init__(self, client):
        super(SqladminV1beta4.BackupRunsService, self).__init__(client)
        self._upload_configs = {}

    def Delete(self, request, global_params=None):
        """Deletes the backup taken by a backup run.

      Args:
        request: (SqlBackupRunsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(http_method='DELETE', method_id='sql.backupRuns.delete', ordered_params=['project', 'instance', 'id'], path_params=['id', 'instance', 'project'], query_params=[], relative_path='sql/v1beta4/projects/{project}/instances/{instance}/backupRuns/{id}', request_field='', request_type_name='SqlBackupRunsDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Retrieves a resource containing information about a backup run.

      Args:
        request: (SqlBackupRunsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (BackupRun) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='sql.backupRuns.get', ordered_params=['project', 'instance', 'id'], path_params=['id', 'instance', 'project'], query_params=[], relative_path='sql/v1beta4/projects/{project}/instances/{instance}/backupRuns/{id}', request_field='', request_type_name='SqlBackupRunsGetRequest', response_type_name='BackupRun', supports_download=False)

    def Insert(self, request, global_params=None):
        """Creates a new backup run on demand.

      Args:
        request: (SqlBackupRunsInsertRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Insert')
        return self._RunMethod(config, request, global_params=global_params)
    Insert.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='sql.backupRuns.insert', ordered_params=['project', 'instance'], path_params=['instance', 'project'], query_params=[], relative_path='sql/v1beta4/projects/{project}/instances/{instance}/backupRuns', request_field='backupRun', request_type_name='SqlBackupRunsInsertRequest', response_type_name='Operation', supports_download=False)

    def List(self, request, global_params=None):
        """Lists all backup runs associated with the project or a given instance and configuration in the reverse chronological order of the backup initiation time.

      Args:
        request: (SqlBackupRunsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (BackupRunsListResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(http_method='GET', method_id='sql.backupRuns.list', ordered_params=['project', 'instance'], path_params=['instance', 'project'], query_params=['maxResults', 'pageToken'], relative_path='sql/v1beta4/projects/{project}/instances/{instance}/backupRuns', request_field='', request_type_name='SqlBackupRunsListRequest', response_type_name='BackupRunsListResponse', supports_download=False)