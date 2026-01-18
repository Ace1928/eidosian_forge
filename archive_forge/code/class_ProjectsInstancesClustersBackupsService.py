from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.bigtableadmin.v2 import bigtableadmin_v2_messages as messages
class ProjectsInstancesClustersBackupsService(base_api.BaseApiService):
    """Service class for the projects_instances_clusters_backups resource."""
    _NAME = 'projects_instances_clusters_backups'

    def __init__(self, client):
        super(BigtableadminV2.ProjectsInstancesClustersBackupsService, self).__init__(client)
        self._upload_configs = {}

    def Copy(self, request, global_params=None):
        """Copy a Cloud Bigtable backup to a new backup in the destination cluster located in the destination instance and project.

      Args:
        request: (BigtableadminProjectsInstancesClustersBackupsCopyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Copy')
        return self._RunMethod(config, request, global_params=global_params)
    Copy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/instances/{instancesId}/clusters/{clustersId}/backups:copy', http_method='POST', method_id='bigtableadmin.projects.instances.clusters.backups.copy', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v2/{+parent}/backups:copy', request_field='copyBackupRequest', request_type_name='BigtableadminProjectsInstancesClustersBackupsCopyRequest', response_type_name='Operation', supports_download=False)

    def Create(self, request, global_params=None):
        """Starts creating a new Cloud Bigtable Backup. The returned backup long-running operation can be used to track creation of the backup. The metadata field type is CreateBackupMetadata. The response field type is Backup, if successful. Cancelling the returned operation will stop the creation and delete the backup.

      Args:
        request: (BigtableadminProjectsInstancesClustersBackupsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/instances/{instancesId}/clusters/{clustersId}/backups', http_method='POST', method_id='bigtableadmin.projects.instances.clusters.backups.create', ordered_params=['parent'], path_params=['parent'], query_params=['backupId'], relative_path='v2/{+parent}/backups', request_field='backup', request_type_name='BigtableadminProjectsInstancesClustersBackupsCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a pending or completed Cloud Bigtable backup.

      Args:
        request: (BigtableadminProjectsInstancesClustersBackupsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/instances/{instancesId}/clusters/{clustersId}/backups/{backupsId}', http_method='DELETE', method_id='bigtableadmin.projects.instances.clusters.backups.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}', request_field='', request_type_name='BigtableadminProjectsInstancesClustersBackupsDeleteRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets metadata on a pending or completed Cloud Bigtable Backup.

      Args:
        request: (BigtableadminProjectsInstancesClustersBackupsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Backup) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/instances/{instancesId}/clusters/{clustersId}/backups/{backupsId}', http_method='GET', method_id='bigtableadmin.projects.instances.clusters.backups.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}', request_field='', request_type_name='BigtableadminProjectsInstancesClustersBackupsGetRequest', response_type_name='Backup', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the access control policy for a Bigtable resource. Returns an empty policy if the resource exists but does not have a policy set.

      Args:
        request: (BigtableadminProjectsInstancesClustersBackupsGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/instances/{instancesId}/clusters/{clustersId}/backups/{backupsId}:getIamPolicy', http_method='POST', method_id='bigtableadmin.projects.instances.clusters.backups.getIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v2/{+resource}:getIamPolicy', request_field='getIamPolicyRequest', request_type_name='BigtableadminProjectsInstancesClustersBackupsGetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def List(self, request, global_params=None):
        """Lists Cloud Bigtable backups. Returns both completed and pending backups.

      Args:
        request: (BigtableadminProjectsInstancesClustersBackupsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListBackupsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/instances/{instancesId}/clusters/{clustersId}/backups', http_method='GET', method_id='bigtableadmin.projects.instances.clusters.backups.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v2/{+parent}/backups', request_field='', request_type_name='BigtableadminProjectsInstancesClustersBackupsListRequest', response_type_name='ListBackupsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates a pending or completed Cloud Bigtable Backup.

      Args:
        request: (BigtableadminProjectsInstancesClustersBackupsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Backup) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/instances/{instancesId}/clusters/{clustersId}/backups/{backupsId}', http_method='PATCH', method_id='bigtableadmin.projects.instances.clusters.backups.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v2/{+name}', request_field='backup', request_type_name='BigtableadminProjectsInstancesClustersBackupsPatchRequest', response_type_name='Backup', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the access control policy on a Bigtable resource. Replaces any existing policy.

      Args:
        request: (BigtableadminProjectsInstancesClustersBackupsSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/instances/{instancesId}/clusters/{clustersId}/backups/{backupsId}:setIamPolicy', http_method='POST', method_id='bigtableadmin.projects.instances.clusters.backups.setIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v2/{+resource}:setIamPolicy', request_field='setIamPolicyRequest', request_type_name='BigtableadminProjectsInstancesClustersBackupsSetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns permissions that the caller has on the specified Bigtable resource.

      Args:
        request: (BigtableadminProjectsInstancesClustersBackupsTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestIamPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/instances/{instancesId}/clusters/{clustersId}/backups/{backupsId}:testIamPermissions', http_method='POST', method_id='bigtableadmin.projects.instances.clusters.backups.testIamPermissions', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v2/{+resource}:testIamPermissions', request_field='testIamPermissionsRequest', request_type_name='BigtableadminProjectsInstancesClustersBackupsTestIamPermissionsRequest', response_type_name='TestIamPermissionsResponse', supports_download=False)