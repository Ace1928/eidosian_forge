from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.vmmigration.v1 import vmmigration_v1_messages as messages
class ProjectsLocationsSourcesMigratingVmsCutoverJobsService(base_api.BaseApiService):
    """Service class for the projects_locations_sources_migratingVms_cutoverJobs resource."""
    _NAME = 'projects_locations_sources_migratingVms_cutoverJobs'

    def __init__(self, client):
        super(VmmigrationV1.ProjectsLocationsSourcesMigratingVmsCutoverJobsService, self).__init__(client)
        self._upload_configs = {}

    def Cancel(self, request, global_params=None):
        """Initiates the cancellation of a running cutover job.

      Args:
        request: (VmmigrationProjectsLocationsSourcesMigratingVmsCutoverJobsCancelRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Cancel')
        return self._RunMethod(config, request, global_params=global_params)
    Cancel.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/sources/{sourcesId}/migratingVms/{migratingVmsId}/cutoverJobs/{cutoverJobsId}:cancel', http_method='POST', method_id='vmmigration.projects.locations.sources.migratingVms.cutoverJobs.cancel', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:cancel', request_field='cancelCutoverJobRequest', request_type_name='VmmigrationProjectsLocationsSourcesMigratingVmsCutoverJobsCancelRequest', response_type_name='Operation', supports_download=False)

    def Create(self, request, global_params=None):
        """Initiates a Cutover of a specific migrating VM. The returned LRO is completed when the cutover job resource is created and the job is initiated.

      Args:
        request: (VmmigrationProjectsLocationsSourcesMigratingVmsCutoverJobsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/sources/{sourcesId}/migratingVms/{migratingVmsId}/cutoverJobs', http_method='POST', method_id='vmmigration.projects.locations.sources.migratingVms.cutoverJobs.create', ordered_params=['parent'], path_params=['parent'], query_params=['cutoverJobId', 'requestId'], relative_path='v1/{+parent}/cutoverJobs', request_field='cutoverJob', request_type_name='VmmigrationProjectsLocationsSourcesMigratingVmsCutoverJobsCreateRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single CutoverJob.

      Args:
        request: (VmmigrationProjectsLocationsSourcesMigratingVmsCutoverJobsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (CutoverJob) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/sources/{sourcesId}/migratingVms/{migratingVmsId}/cutoverJobs/{cutoverJobsId}', http_method='GET', method_id='vmmigration.projects.locations.sources.migratingVms.cutoverJobs.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='VmmigrationProjectsLocationsSourcesMigratingVmsCutoverJobsGetRequest', response_type_name='CutoverJob', supports_download=False)

    def List(self, request, global_params=None):
        """Lists the CutoverJobs of a migrating VM. Only 25 most recent CutoverJobs are listed.

      Args:
        request: (VmmigrationProjectsLocationsSourcesMigratingVmsCutoverJobsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListCutoverJobsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/locations/{locationsId}/sources/{sourcesId}/migratingVms/{migratingVmsId}/cutoverJobs', http_method='GET', method_id='vmmigration.projects.locations.sources.migratingVms.cutoverJobs.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/cutoverJobs', request_field='', request_type_name='VmmigrationProjectsLocationsSourcesMigratingVmsCutoverJobsListRequest', response_type_name='ListCutoverJobsResponse', supports_download=False)