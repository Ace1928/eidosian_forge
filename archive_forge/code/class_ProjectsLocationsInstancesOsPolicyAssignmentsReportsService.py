from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.osconfig.v1alpha import osconfig_v1alpha_messages as messages
class ProjectsLocationsInstancesOsPolicyAssignmentsReportsService(base_api.BaseApiService):
    """Service class for the projects_locations_instances_osPolicyAssignments_reports resource."""
    _NAME = 'projects_locations_instances_osPolicyAssignments_reports'

    def __init__(self, client):
        super(OsconfigV1alpha.ProjectsLocationsInstancesOsPolicyAssignmentsReportsService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Get the OS policy assignment report for the specified Compute Engine VM instance.

      Args:
        request: (OsconfigProjectsLocationsInstancesOsPolicyAssignmentsReportsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (OSPolicyAssignmentReport) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/instances/{instancesId}/osPolicyAssignments/{osPolicyAssignmentsId}/report', http_method='GET', method_id='osconfig.projects.locations.instances.osPolicyAssignments.reports.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha/{+name}', request_field='', request_type_name='OsconfigProjectsLocationsInstancesOsPolicyAssignmentsReportsGetRequest', response_type_name='OSPolicyAssignmentReport', supports_download=False)

    def List(self, request, global_params=None):
        """List OS policy assignment reports for all Compute Engine VM instances in the specified zone.

      Args:
        request: (OsconfigProjectsLocationsInstancesOsPolicyAssignmentsReportsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListOSPolicyAssignmentReportsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/instances/{instancesId}/osPolicyAssignments/{osPolicyAssignmentsId}/reports', http_method='GET', method_id='osconfig.projects.locations.instances.osPolicyAssignments.reports.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v1alpha/{+parent}/reports', request_field='', request_type_name='OsconfigProjectsLocationsInstancesOsPolicyAssignmentsReportsListRequest', response_type_name='ListOSPolicyAssignmentReportsResponse', supports_download=False)