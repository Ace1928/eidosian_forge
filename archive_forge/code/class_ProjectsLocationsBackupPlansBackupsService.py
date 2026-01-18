from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.gkebackup.v1 import gkebackup_v1_messages as messages
class ProjectsLocationsBackupPlansBackupsService(base_api.BaseApiService):
    """Service class for the projects_locations_backupPlans_backups resource."""
    _NAME = 'projects_locations_backupPlans_backups'

    def __init__(self, client):
        super(GkebackupV1.ProjectsLocationsBackupPlansBackupsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a Backup for the given BackupPlan.

      Args:
        request: (GkebackupProjectsLocationsBackupPlansBackupsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/backupPlans/{backupPlansId}/backups', http_method='POST', method_id='gkebackup.projects.locations.backupPlans.backups.create', ordered_params=['parent'], path_params=['parent'], query_params=['backupId'], relative_path='v1/{+parent}/backups', request_field='backup', request_type_name='GkebackupProjectsLocationsBackupPlansBackupsCreateRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes an existing Backup.

      Args:
        request: (GkebackupProjectsLocationsBackupPlansBackupsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/backupPlans/{backupPlansId}/backups/{backupsId}', http_method='DELETE', method_id='gkebackup.projects.locations.backupPlans.backups.delete', ordered_params=['name'], path_params=['name'], query_params=['etag', 'force'], relative_path='v1/{+name}', request_field='', request_type_name='GkebackupProjectsLocationsBackupPlansBackupsDeleteRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def Get(self, request, global_params=None):
        """Retrieve the details of a single Backup.

      Args:
        request: (GkebackupProjectsLocationsBackupPlansBackupsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Backup) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/backupPlans/{backupPlansId}/backups/{backupsId}', http_method='GET', method_id='gkebackup.projects.locations.backupPlans.backups.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='GkebackupProjectsLocationsBackupPlansBackupsGetRequest', response_type_name='Backup', supports_download=False)

    def GetBackupIndexDownloadUrl(self, request, global_params=None):
        """Retrieve the link to the backupIndex.

      Args:
        request: (GkebackupProjectsLocationsBackupPlansBackupsGetBackupIndexDownloadUrlRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GetBackupIndexDownloadUrlResponse) The response message.
      """
        config = self.GetMethodConfig('GetBackupIndexDownloadUrl')
        return self._RunMethod(config, request, global_params=global_params)
    GetBackupIndexDownloadUrl.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/backupPlans/{backupPlansId}/backups/{backupsId}:getBackupIndexDownloadUrl', http_method='GET', method_id='gkebackup.projects.locations.backupPlans.backups.getBackupIndexDownloadUrl', ordered_params=['backup'], path_params=['backup'], query_params=[], relative_path='v1/{+backup}:getBackupIndexDownloadUrl', request_field='', request_type_name='GkebackupProjectsLocationsBackupPlansBackupsGetBackupIndexDownloadUrlRequest', response_type_name='GetBackupIndexDownloadUrlResponse', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the access control policy for a resource. Returns an empty policy if the resource exists and does not have a policy set.

      Args:
        request: (GkebackupProjectsLocationsBackupPlansBackupsGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/backupPlans/{backupPlansId}/backups/{backupsId}:getIamPolicy', http_method='GET', method_id='gkebackup.projects.locations.backupPlans.backups.getIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=['options_requestedPolicyVersion'], relative_path='v1/{+resource}:getIamPolicy', request_field='', request_type_name='GkebackupProjectsLocationsBackupPlansBackupsGetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def List(self, request, global_params=None):
        """Lists the Backups for a given BackupPlan.

      Args:
        request: (GkebackupProjectsLocationsBackupPlansBackupsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListBackupsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/backupPlans/{backupPlansId}/backups', http_method='GET', method_id='gkebackup.projects.locations.backupPlans.backups.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/backups', request_field='', request_type_name='GkebackupProjectsLocationsBackupPlansBackupsListRequest', response_type_name='ListBackupsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Update a Backup.

      Args:
        request: (GkebackupProjectsLocationsBackupPlansBackupsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/backupPlans/{backupPlansId}/backups/{backupsId}', http_method='PATCH', method_id='gkebackup.projects.locations.backupPlans.backups.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1/{+name}', request_field='backup', request_type_name='GkebackupProjectsLocationsBackupPlansBackupsPatchRequest', response_type_name='GoogleLongrunningOperation', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the access control policy on the specified resource. Replaces any existing policy. Can return `NOT_FOUND`, `INVALID_ARGUMENT`, and `PERMISSION_DENIED` errors.

      Args:
        request: (GkebackupProjectsLocationsBackupPlansBackupsSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/backupPlans/{backupPlansId}/backups/{backupsId}:setIamPolicy', http_method='POST', method_id='gkebackup.projects.locations.backupPlans.backups.setIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:setIamPolicy', request_field='setIamPolicyRequest', request_type_name='GkebackupProjectsLocationsBackupPlansBackupsSetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that a caller has on the specified resource. If the resource does not exist, this will return an empty set of permissions, not a `NOT_FOUND` error. Note: This operation is designed to be used for building permission-aware UIs and command-line tools, not for authorization checking. This operation may "fail open" without warning.

      Args:
        request: (GkebackupProjectsLocationsBackupPlansBackupsTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestIamPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/backupPlans/{backupPlansId}/backups/{backupsId}:testIamPermissions', http_method='POST', method_id='gkebackup.projects.locations.backupPlans.backups.testIamPermissions', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:testIamPermissions', request_field='testIamPermissionsRequest', request_type_name='GkebackupProjectsLocationsBackupPlansBackupsTestIamPermissionsRequest', response_type_name='TestIamPermissionsResponse', supports_download=False)