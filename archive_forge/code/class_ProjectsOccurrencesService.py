from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.containeranalysis.v1 import containeranalysis_v1_messages as messages
class ProjectsOccurrencesService(base_api.BaseApiService):
    """Service class for the projects_occurrences resource."""
    _NAME = 'projects_occurrences'

    def __init__(self, client):
        super(ContaineranalysisV1.ProjectsOccurrencesService, self).__init__(client)
        self._upload_configs = {}

    def BatchCreate(self, request, global_params=None):
        """Creates new occurrences in batch.

      Args:
        request: (ContaineranalysisProjectsOccurrencesBatchCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (BatchCreateOccurrencesResponse) The response message.
      """
        config = self.GetMethodConfig('BatchCreate')
        return self._RunMethod(config, request, global_params=global_params)
    BatchCreate.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/occurrences:batchCreate', http_method='POST', method_id='containeranalysis.projects.occurrences.batchCreate', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/occurrences:batchCreate', request_field='batchCreateOccurrencesRequest', request_type_name='ContaineranalysisProjectsOccurrencesBatchCreateRequest', response_type_name='BatchCreateOccurrencesResponse', supports_download=False)

    def Create(self, request, global_params=None):
        """Creates a new occurrence.

      Args:
        request: (ContaineranalysisProjectsOccurrencesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Occurrence) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/occurrences', http_method='POST', method_id='containeranalysis.projects.occurrences.create', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/occurrences', request_field='occurrence', request_type_name='ContaineranalysisProjectsOccurrencesCreateRequest', response_type_name='Occurrence', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes the specified occurrence. For example, use this method to delete an occurrence when the occurrence is no longer applicable for the given resource.

      Args:
        request: (ContaineranalysisProjectsOccurrencesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/occurrences/{occurrencesId}', http_method='DELETE', method_id='containeranalysis.projects.occurrences.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ContaineranalysisProjectsOccurrencesDeleteRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets the specified occurrence.

      Args:
        request: (ContaineranalysisProjectsOccurrencesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Occurrence) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/occurrences/{occurrencesId}', http_method='GET', method_id='containeranalysis.projects.occurrences.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ContaineranalysisProjectsOccurrencesGetRequest', response_type_name='Occurrence', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the access control policy for a note or an occurrence resource. Requires `containeranalysis.notes.setIamPolicy` or `containeranalysis.occurrences.setIamPolicy` permission if the resource is a note or occurrence, respectively. The resource takes the format `projects/[PROJECT_ID]/notes/[NOTE_ID]` for notes and `projects/[PROJECT_ID]/occurrences/[OCCURRENCE_ID]` for occurrences.

      Args:
        request: (ContaineranalysisProjectsOccurrencesGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/occurrences/{occurrencesId}:getIamPolicy', http_method='POST', method_id='containeranalysis.projects.occurrences.getIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:getIamPolicy', request_field='getIamPolicyRequest', request_type_name='ContaineranalysisProjectsOccurrencesGetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def GetNotes(self, request, global_params=None):
        """Gets the note attached to the specified occurrence. Consumer projects can use this method to get a note that belongs to a provider project.

      Args:
        request: (ContaineranalysisProjectsOccurrencesGetNotesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Note) The response message.
      """
        config = self.GetMethodConfig('GetNotes')
        return self._RunMethod(config, request, global_params=global_params)
    GetNotes.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/occurrences/{occurrencesId}/notes', http_method='GET', method_id='containeranalysis.projects.occurrences.getNotes', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}/notes', request_field='', request_type_name='ContaineranalysisProjectsOccurrencesGetNotesRequest', response_type_name='Note', supports_download=False)

    def GetVulnerabilitySummary(self, request, global_params=None):
        """Gets a summary of the number and severity of occurrences.

      Args:
        request: (ContaineranalysisProjectsOccurrencesGetVulnerabilitySummaryRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (VulnerabilityOccurrencesSummary) The response message.
      """
        config = self.GetMethodConfig('GetVulnerabilitySummary')
        return self._RunMethod(config, request, global_params=global_params)
    GetVulnerabilitySummary.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/occurrences:vulnerabilitySummary', http_method='GET', method_id='containeranalysis.projects.occurrences.getVulnerabilitySummary', ordered_params=['parent'], path_params=['parent'], query_params=['filter'], relative_path='v1/{+parent}/occurrences:vulnerabilitySummary', request_field='', request_type_name='ContaineranalysisProjectsOccurrencesGetVulnerabilitySummaryRequest', response_type_name='VulnerabilityOccurrencesSummary', supports_download=False)

    def List(self, request, global_params=None):
        """Lists occurrences for the specified project.

      Args:
        request: (ContaineranalysisProjectsOccurrencesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListOccurrencesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/occurrences', http_method='GET', method_id='containeranalysis.projects.occurrences.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/occurrences', request_field='', request_type_name='ContaineranalysisProjectsOccurrencesListRequest', response_type_name='ListOccurrencesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the specified occurrence.

      Args:
        request: (ContaineranalysisProjectsOccurrencesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Occurrence) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/occurrences/{occurrencesId}', http_method='PATCH', method_id='containeranalysis.projects.occurrences.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1/{+name}', request_field='occurrence', request_type_name='ContaineranalysisProjectsOccurrencesPatchRequest', response_type_name='Occurrence', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the access control policy on the specified note or occurrence. Requires `containeranalysis.notes.setIamPolicy` or `containeranalysis.occurrences.setIamPolicy` permission if the resource is a note or an occurrence, respectively. The resource takes the format `projects/[PROJECT_ID]/notes/[NOTE_ID]` for notes and `projects/[PROJECT_ID]/occurrences/[OCCURRENCE_ID]` for occurrences.

      Args:
        request: (ContaineranalysisProjectsOccurrencesSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/occurrences/{occurrencesId}:setIamPolicy', http_method='POST', method_id='containeranalysis.projects.occurrences.setIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:setIamPolicy', request_field='setIamPolicyRequest', request_type_name='ContaineranalysisProjectsOccurrencesSetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns the permissions that a caller has on the specified note or occurrence. Requires list permission on the project (for example, `containeranalysis.notes.list`). The resource takes the format `projects/[PROJECT_ID]/notes/[NOTE_ID]` for notes and `projects/[PROJECT_ID]/occurrences/[OCCURRENCE_ID]` for occurrences.

      Args:
        request: (ContaineranalysisProjectsOccurrencesTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestIamPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/occurrences/{occurrencesId}:testIamPermissions', http_method='POST', method_id='containeranalysis.projects.occurrences.testIamPermissions', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:testIamPermissions', request_field='testIamPermissionsRequest', request_type_name='ContaineranalysisProjectsOccurrencesTestIamPermissionsRequest', response_type_name='TestIamPermissionsResponse', supports_download=False)