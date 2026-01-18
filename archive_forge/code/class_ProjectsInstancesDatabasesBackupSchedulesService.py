from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.spanner.v1 import spanner_v1_messages as messages
class ProjectsInstancesDatabasesBackupSchedulesService(base_api.BaseApiService):
    """Service class for the projects_instances_databases_backupSchedules resource."""
    _NAME = 'projects_instances_databases_backupSchedules'

    def __init__(self, client):
        super(SpannerV1.ProjectsInstancesDatabasesBackupSchedulesService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new backup schedule.

      Args:
        request: (SpannerProjectsInstancesDatabasesBackupSchedulesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (BackupSchedule) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/instances/{instancesId}/databases/{databasesId}/backupSchedules', http_method='POST', method_id='spanner.projects.instances.databases.backupSchedules.create', ordered_params=['parent'], path_params=['parent'], query_params=['backupScheduleId'], relative_path='v1/{+parent}/backupSchedules', request_field='backupSchedule', request_type_name='SpannerProjectsInstancesDatabasesBackupSchedulesCreateRequest', response_type_name='BackupSchedule', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a backup schedule.

      Args:
        request: (SpannerProjectsInstancesDatabasesBackupSchedulesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/instances/{instancesId}/databases/{databasesId}/backupSchedules/{backupSchedulesId}', http_method='DELETE', method_id='spanner.projects.instances.databases.backupSchedules.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='SpannerProjectsInstancesDatabasesBackupSchedulesDeleteRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets backup schedule for the input schedule name.

      Args:
        request: (SpannerProjectsInstancesDatabasesBackupSchedulesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (BackupSchedule) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/instances/{instancesId}/databases/{databasesId}/backupSchedules/{backupSchedulesId}', http_method='GET', method_id='spanner.projects.instances.databases.backupSchedules.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='SpannerProjectsInstancesDatabasesBackupSchedulesGetRequest', response_type_name='BackupSchedule', supports_download=False)

    def List(self, request, global_params=None):
        """Lists all the backup schedules for the database.

      Args:
        request: (SpannerProjectsInstancesDatabasesBackupSchedulesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListBackupSchedulesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/instances/{instancesId}/databases/{databasesId}/backupSchedules', http_method='GET', method_id='spanner.projects.instances.databases.backupSchedules.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1/{+parent}/backupSchedules', request_field='', request_type_name='SpannerProjectsInstancesDatabasesBackupSchedulesListRequest', response_type_name='ListBackupSchedulesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates a backup schedule.

      Args:
        request: (SpannerProjectsInstancesDatabasesBackupSchedulesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (BackupSchedule) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/instances/{instancesId}/databases/{databasesId}/backupSchedules/{backupSchedulesId}', http_method='PATCH', method_id='spanner.projects.instances.databases.backupSchedules.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1/{+name}', request_field='backupSchedule', request_type_name='SpannerProjectsInstancesDatabasesBackupSchedulesPatchRequest', response_type_name='BackupSchedule', supports_download=False)