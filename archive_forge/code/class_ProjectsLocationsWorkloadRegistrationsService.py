from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.workloadcertificate.v1alpha1 import workloadcertificate_v1alpha1_messages as messages
class ProjectsLocationsWorkloadRegistrationsService(base_api.BaseApiService):
    """Service class for the projects_locations_workloadRegistrations resource."""
    _NAME = 'projects_locations_workloadRegistrations'

    def __init__(self, client):
        super(WorkloadcertificateV1alpha1.ProjectsLocationsWorkloadRegistrationsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Creates a new WorkloadRegistration in a given project and location.

      Args:
        request: (WorkloadcertificateProjectsLocationsWorkloadRegistrationsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/workloadRegistrations', http_method='POST', method_id='workloadcertificate.projects.locations.workloadRegistrations.create', ordered_params=['parent'], path_params=['parent'], query_params=['requestId', 'workloadRegistrationId'], relative_path='v1alpha1/{+parent}/workloadRegistrations', request_field='workloadRegistration', request_type_name='WorkloadcertificateProjectsLocationsWorkloadRegistrationsCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes a single WorkloadRegistration.

      Args:
        request: (WorkloadcertificateProjectsLocationsWorkloadRegistrationsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/workloadRegistrations/{workloadRegistrationsId}', http_method='DELETE', method_id='workloadcertificate.projects.locations.workloadRegistrations.delete', ordered_params=['name'], path_params=['name'], query_params=['requestId'], relative_path='v1alpha1/{+name}', request_field='', request_type_name='WorkloadcertificateProjectsLocationsWorkloadRegistrationsDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets details of a single WorkloadRegistration.

      Args:
        request: (WorkloadcertificateProjectsLocationsWorkloadRegistrationsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (WorkloadRegistration) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/workloadRegistrations/{workloadRegistrationsId}', http_method='GET', method_id='workloadcertificate.projects.locations.workloadRegistrations.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha1/{+name}', request_field='', request_type_name='WorkloadcertificateProjectsLocationsWorkloadRegistrationsGetRequest', response_type_name='WorkloadRegistration', supports_download=False)

    def List(self, request, global_params=None):
        """Lists WorkloadRegistrations in a given project and location.

      Args:
        request: (WorkloadcertificateProjectsLocationsWorkloadRegistrationsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListWorkloadRegistrationsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha1/projects/{projectsId}/locations/{locationsId}/workloadRegistrations', http_method='GET', method_id='workloadcertificate.projects.locations.workloadRegistrations.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v1alpha1/{+parent}/workloadRegistrations', request_field='', request_type_name='WorkloadcertificateProjectsLocationsWorkloadRegistrationsListRequest', response_type_name='ListWorkloadRegistrationsResponse', supports_download=False)