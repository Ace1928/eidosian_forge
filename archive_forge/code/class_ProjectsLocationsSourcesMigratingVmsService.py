from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.vmmigration.v1 import vmmigration_v1_messages as messages
class ProjectsLocationsSourcesMigratingVmsService(base_api.BaseApiService):
    """Service class for the projects_locations_sources_migratingVms resource."""
    _NAME = 'projects_locations_sources_migratingVms'

    def __init__(self, client):
        super(VmmigrationV1.ProjectsLocationsSourcesMigratingVmsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new MigratingVm in a given Source.

      Args:
        request: (VmmigrationProjectsLocationsSourcesMigratingVmsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/sources/{sourcesId}/migratingVms', http_method='POST', method_id='vmmigration.projects.locations.sources.migratingVms.create', ordered_params=['parent'], path_params=['parent'], query_params=['migratingVmId', 'requestId'], relative_path='v1/{+parent}/migratingVms', request_field='migratingVm', request_type_name='VmmigrationProjectsLocationsSourcesMigratingVmsCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a single MigratingVm.

      Args:
        request: (VmmigrationProjectsLocationsSourcesMigratingVmsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/sources/{sourcesId}/migratingVms/{migratingVmsId}', http_method='DELETE', method_id='vmmigration.projects.locations.sources.migratingVms.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='VmmigrationProjectsLocationsSourcesMigratingVmsDeleteRequest', response_type_name='Operation', supports_download=False)

    def FinalizeMigration(self, request, global_params=None):
        """Marks a migration as completed, deleting migration resources that are no longer being used. Only applicable after cutover is done.

      Args:
        request: (VmmigrationProjectsLocationsSourcesMigratingVmsFinalizeMigrationRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('FinalizeMigration')
        return self._RunMethod(config, request, global_params=global_params)
    FinalizeMigration.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/sources/{sourcesId}/migratingVms/{migratingVmsId}:finalizeMigration', http_method='POST', method_id='vmmigration.projects.locations.sources.migratingVms.finalizeMigration', ordered_params=['migratingVm'], path_params=['migratingVm'], query_params=[], relative_path='v1/{+migratingVm}:finalizeMigration', request_field='finalizeMigrationRequest', request_type_name='VmmigrationProjectsLocationsSourcesMigratingVmsFinalizeMigrationRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single MigratingVm.

      Args:
        request: (VmmigrationProjectsLocationsSourcesMigratingVmsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (MigratingVm) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/sources/{sourcesId}/migratingVms/{migratingVmsId}', http_method='GET', method_id='vmmigration.projects.locations.sources.migratingVms.get', ordered_params=['name'], path_params=['name'], query_params=['view'], relative_path='v1/{+name}', request_field='', request_type_name='VmmigrationProjectsLocationsSourcesMigratingVmsGetRequest', response_type_name='MigratingVm', supports_download=False)

    def List(self, request, global_params=None):
        """Lists MigratingVms in a given Source.

      Args:
        request: (VmmigrationProjectsLocationsSourcesMigratingVmsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListMigratingVmsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/sources/{sourcesId}/migratingVms', http_method='GET', method_id='vmmigration.projects.locations.sources.migratingVms.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken', 'view'], relative_path='v1/{+parent}/migratingVms', request_field='', request_type_name='VmmigrationProjectsLocationsSourcesMigratingVmsListRequest', response_type_name='ListMigratingVmsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the parameters of a single MigratingVm.

      Args:
        request: (VmmigrationProjectsLocationsSourcesMigratingVmsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/sources/{sourcesId}/migratingVms/{migratingVmsId}', http_method='PATCH', method_id='vmmigration.projects.locations.sources.migratingVms.patch', ordered_params=['name'], path_params=['name'], query_params=['requestId', 'updateMask'], relative_path='v1/{+name}', request_field='migratingVm', request_type_name='VmmigrationProjectsLocationsSourcesMigratingVmsPatchRequest', response_type_name='Operation', supports_download=False)

    def PauseMigration(self, request, global_params=None):
        """Pauses a migration for a VM. If cycle tasks are running they will be cancelled, preserving source task data. Further replication cycles will not be triggered while the VM is paused.

      Args:
        request: (VmmigrationProjectsLocationsSourcesMigratingVmsPauseMigrationRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('PauseMigration')
        return self._RunMethod(config, request, global_params=global_params)
    PauseMigration.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/sources/{sourcesId}/migratingVms/{migratingVmsId}:pauseMigration', http_method='POST', method_id='vmmigration.projects.locations.sources.migratingVms.pauseMigration', ordered_params=['migratingVm'], path_params=['migratingVm'], query_params=[], relative_path='v1/{+migratingVm}:pauseMigration', request_field='pauseMigrationRequest', request_type_name='VmmigrationProjectsLocationsSourcesMigratingVmsPauseMigrationRequest', response_type_name='Operation', supports_download=False)

    def ResumeMigration(self, request, global_params=None):
        """Resumes a migration for a VM. When called on a paused migration, will start the process of uploading data and creating snapshots; when called on a completed cut-over migration, will update the migration to active state and start the process of uploading data and creating snapshots.

      Args:
        request: (VmmigrationProjectsLocationsSourcesMigratingVmsResumeMigrationRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('ResumeMigration')
        return self._RunMethod(config, request, global_params=global_params)
    ResumeMigration.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/sources/{sourcesId}/migratingVms/{migratingVmsId}:resumeMigration', http_method='POST', method_id='vmmigration.projects.locations.sources.migratingVms.resumeMigration', ordered_params=['migratingVm'], path_params=['migratingVm'], query_params=[], relative_path='v1/{+migratingVm}:resumeMigration', request_field='resumeMigrationRequest', request_type_name='VmmigrationProjectsLocationsSourcesMigratingVmsResumeMigrationRequest', response_type_name='Operation', supports_download=False)

    def StartMigration(self, request, global_params=None):
        """Starts migration for a VM. Starts the process of uploading data and creating snapshots, in replication cycles scheduled by the policy.

      Args:
        request: (VmmigrationProjectsLocationsSourcesMigratingVmsStartMigrationRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('StartMigration')
        return self._RunMethod(config, request, global_params=global_params)
    StartMigration.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/sources/{sourcesId}/migratingVms/{migratingVmsId}:startMigration', http_method='POST', method_id='vmmigration.projects.locations.sources.migratingVms.startMigration', ordered_params=['migratingVm'], path_params=['migratingVm'], query_params=[], relative_path='v1/{+migratingVm}:startMigration', request_field='startMigrationRequest', request_type_name='VmmigrationProjectsLocationsSourcesMigratingVmsStartMigrationRequest', response_type_name='Operation', supports_download=False)