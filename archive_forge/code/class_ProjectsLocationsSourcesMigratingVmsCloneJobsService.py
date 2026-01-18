from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.vmmigration.v1 import vmmigration_v1_messages as messages
class ProjectsLocationsSourcesMigratingVmsCloneJobsService(base_api.BaseApiService):
    """Service class for the projects_locations_sources_migratingVms_cloneJobs resource."""
    _NAME = 'projects_locations_sources_migratingVms_cloneJobs'

    def __init__(self, client):
        super(VmmigrationV1.ProjectsLocationsSourcesMigratingVmsCloneJobsService, self).__init__(client)
        self._upload_configs = {}

    def Cancel(self, request, global_params=None):
        """Initiates the cancellation of a running clone job.

      Args:
        request: (VmmigrationProjectsLocationsSourcesMigratingVmsCloneJobsCancelRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Cancel')
        return self._RunMethod(config, request, global_params=global_params)
    Cancel.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/sources/{sourcesId}/migratingVms/{migratingVmsId}/cloneJobs/{cloneJobsId}:cancel', http_method='POST', method_id='vmmigration.projects.locations.sources.migratingVms.cloneJobs.cancel', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:cancel', request_field='cancelCloneJobRequest', request_type_name='VmmigrationProjectsLocationsSourcesMigratingVmsCloneJobsCancelRequest', response_type_name='Operation', supports_download=False)

    def Create(self, request, global_params=None):
        """Initiates a Clone of a specific migrating VM.

      Args:
        request: (VmmigrationProjectsLocationsSourcesMigratingVmsCloneJobsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/sources/{sourcesId}/migratingVms/{migratingVmsId}/cloneJobs', http_method='POST', method_id='vmmigration.projects.locations.sources.migratingVms.cloneJobs.create', ordered_params=['parent'], path_params=['parent'], query_params=['cloneJobId', 'requestId'], relative_path='v1/{+parent}/cloneJobs', request_field='cloneJob', request_type_name='VmmigrationProjectsLocationsSourcesMigratingVmsCloneJobsCreateRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single CloneJob.

      Args:
        request: (VmmigrationProjectsLocationsSourcesMigratingVmsCloneJobsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (CloneJob) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/sources/{sourcesId}/migratingVms/{migratingVmsId}/cloneJobs/{cloneJobsId}', http_method='GET', method_id='vmmigration.projects.locations.sources.migratingVms.cloneJobs.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='VmmigrationProjectsLocationsSourcesMigratingVmsCloneJobsGetRequest', response_type_name='CloneJob', supports_download=False)

    def List(self, request, global_params=None):
        """Lists the CloneJobs of a migrating VM. Only 25 most recent CloneJobs are listed.

      Args:
        request: (VmmigrationProjectsLocationsSourcesMigratingVmsCloneJobsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListCloneJobsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/sources/{sourcesId}/migratingVms/{migratingVmsId}/cloneJobs', http_method='GET', method_id='vmmigration.projects.locations.sources.migratingVms.cloneJobs.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/cloneJobs', request_field='', request_type_name='VmmigrationProjectsLocationsSourcesMigratingVmsCloneJobsListRequest', response_type_name='ListCloneJobsResponse', supports_download=False)