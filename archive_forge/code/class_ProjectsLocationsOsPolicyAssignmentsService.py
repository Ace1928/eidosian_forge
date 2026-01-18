from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.osconfig.v1alpha import osconfig_v1alpha_messages as messages
class ProjectsLocationsOsPolicyAssignmentsService(base_api.BaseApiService):
    """Service class for the projects_locations_osPolicyAssignments resource."""
    _NAME = 'projects_locations_osPolicyAssignments'

    def __init__(self, client):
        super(OsconfigV1alpha.ProjectsLocationsOsPolicyAssignmentsService, self).__init__(client)
        self._upload_configs = {}

    def Create(self, request, global_params=None):
        """Create an OS policy assignment. This method also creates the first revision of the OS policy assignment. This method returns a long running operation (LRO) that contains the rollout details. The rollout can be cancelled by cancelling the LRO. For more information, see [Method: projects.locations.osPolicyAssignments.operations.cancel](https://cloud.google.com/compute/docs/osconfig/rest/v1alpha/projects.locations.osPolicyAssignments.operations/cancel).

      Args:
        request: (OsconfigProjectsLocationsOsPolicyAssignmentsCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/osPolicyAssignments', http_method='POST', method_id='osconfig.projects.locations.osPolicyAssignments.create', ordered_params=['parent'], path_params=['parent'], query_params=['osPolicyAssignmentId'], relative_path='v1alpha/{+parent}/osPolicyAssignments', request_field='oSPolicyAssignment', request_type_name='OsconfigProjectsLocationsOsPolicyAssignmentsCreateRequest', response_type_name='Operation', supports_download=False)

    def Delete(self, request, global_params=None):
        """Delete the OS policy assignment. This method creates a new revision of the OS policy assignment. This method returns a long running operation (LRO) that contains the rollout details. The rollout can be cancelled by cancelling the LRO. If the LRO completes and is not cancelled, all revisions associated with the OS policy assignment are deleted. For more information, see [Method: projects.locations.osPolicyAssignments.operations.cancel](https://cloud.google.com/compute/docs/osconfig/rest/v1alpha/projects.locations.osPolicyAssignments.operations/cancel).

      Args:
        request: (OsconfigProjectsLocationsOsPolicyAssignmentsDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/osPolicyAssignments/{osPolicyAssignmentsId}', http_method='DELETE', method_id='osconfig.projects.locations.osPolicyAssignments.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha/{+name}', request_field='', request_type_name='OsconfigProjectsLocationsOsPolicyAssignmentsDeleteRequest', response_type_name='Operation', supports_download=False)

    def Get(self, request, global_params=None):
        """Retrieve an existing OS policy assignment. This method always returns the latest revision. In order to retrieve a previous revision of the assignment, also provide the revision ID in the `name` parameter.

      Args:
        request: (OsconfigProjectsLocationsOsPolicyAssignmentsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (OSPolicyAssignment) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/osPolicyAssignments/{osPolicyAssignmentsId}', http_method='GET', method_id='osconfig.projects.locations.osPolicyAssignments.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha/{+name}', request_field='', request_type_name='OsconfigProjectsLocationsOsPolicyAssignmentsGetRequest', response_type_name='OSPolicyAssignment', supports_download=False)

    def List(self, request, global_params=None):
        """List the OS policy assignments under the parent resource. For each OS policy assignment, the latest revision is returned.

      Args:
        request: (OsconfigProjectsLocationsOsPolicyAssignmentsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListOSPolicyAssignmentsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/osPolicyAssignments', http_method='GET', method_id='osconfig.projects.locations.osPolicyAssignments.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1alpha/{+parent}/osPolicyAssignments', request_field='', request_type_name='OsconfigProjectsLocationsOsPolicyAssignmentsListRequest', response_type_name='ListOSPolicyAssignmentsResponse', supports_download=False)

    def ListRevisions(self, request, global_params=None):
        """List the OS policy assignment revisions for a given OS policy assignment.

      Args:
        request: (OsconfigProjectsLocationsOsPolicyAssignmentsListRevisionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListOSPolicyAssignmentRevisionsResponse) The response message.
      """
        config = self.GetMethodConfig('ListRevisions')
        return self._RunMethod(config, request, global_params=global_params)
    ListRevisions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/osPolicyAssignments/{osPolicyAssignmentsId}:listRevisions', http_method='GET', method_id='osconfig.projects.locations.osPolicyAssignments.listRevisions', ordered_params=['name'], path_params=['name'], query_params=['pageSize', 'pageToken'], relative_path='v1alpha/{+name}:listRevisions', request_field='', request_type_name='OsconfigProjectsLocationsOsPolicyAssignmentsListRevisionsRequest', response_type_name='ListOSPolicyAssignmentRevisionsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Update an existing OS policy assignment. This method creates a new revision of the OS policy assignment. This method returns a long running operation (LRO) that contains the rollout details. The rollout can be cancelled by cancelling the LRO. For more information, see [Method: projects.locations.osPolicyAssignments.operations.cancel](https://cloud.google.com/compute/docs/osconfig/rest/v1alpha/projects.locations.osPolicyAssignments.operations/cancel).

      Args:
        request: (OsconfigProjectsLocationsOsPolicyAssignmentsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/locations/{locationsId}/osPolicyAssignments/{osPolicyAssignmentsId}', http_method='PATCH', method_id='osconfig.projects.locations.osPolicyAssignments.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1alpha/{+name}', request_field='oSPolicyAssignment', request_type_name='OsconfigProjectsLocationsOsPolicyAssignmentsPatchRequest', response_type_name='Operation', supports_download=False)