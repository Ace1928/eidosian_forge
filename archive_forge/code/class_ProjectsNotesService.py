from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.containeranalysis.v1 import containeranalysis_v1_messages as messages
class ProjectsNotesService(base_api.BaseApiService):
    """Service class for the projects_notes resource."""
    _NAME = 'projects_notes'

    def __init__(self, client):
        super(ContaineranalysisV1.ProjectsNotesService, self).__init__(client)
        self._upload_configs = {}

    def BatchCreate(self, request, global_params=None):
        """Creates new notes in batch.

      Args:
        request: (ContaineranalysisProjectsNotesBatchCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (BatchCreateNotesResponse) The response message.
      """
        config = self.GetMethodConfig('BatchCreate')
        return self._RunMethod(config, request, global_params=global_params)
    BatchCreate.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/notes:batchCreate', http_method='POST', method_id='containeranalysis.projects.notes.batchCreate', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v1/{+parent}/notes:batchCreate', request_field='batchCreateNotesRequest', request_type_name='ContaineranalysisProjectsNotesBatchCreateRequest', response_type_name='BatchCreateNotesResponse', supports_download=False)

    def Create(self, request, global_params=None):
        """Creates a new note.

      Args:
        request: (ContaineranalysisProjectsNotesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Note) The response message.
      """
        config = self.GetMethodConfig('Create')
        return self._RunMethod(config, request, global_params=global_params)
    Create.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/notes', http_method='POST', method_id='containeranalysis.projects.notes.create', ordered_params=['parent'], path_params=['parent'], query_params=['noteId'], relative_path='v1/{+parent}/notes', request_field='note', request_type_name='ContaineranalysisProjectsNotesCreateRequest', response_type_name='Note', supports_download=False)

    def Delete(self, request, global_params=None):
        """Deletes the specified note.

      Args:
        request: (ContaineranalysisProjectsNotesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
        config = self.GetMethodConfig('Delete')
        return self._RunMethod(config, request, global_params=global_params)
    Delete.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/notes/{notesId}', http_method='DELETE', method_id='containeranalysis.projects.notes.delete', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ContaineranalysisProjectsNotesDeleteRequest', response_type_name='Empty', supports_download=False)

    def Get(self, request, global_params=None):
        """Gets the specified note.

      Args:
        request: (ContaineranalysisProjectsNotesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Note) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/notes/{notesId}', http_method='GET', method_id='containeranalysis.projects.notes.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}', request_field='', request_type_name='ContaineranalysisProjectsNotesGetRequest', response_type_name='Note', supports_download=False)

    def GetIamPolicy(self, request, global_params=None):
        """Gets the access control policy for a note or an occurrence resource. Requires `containeranalysis.notes.setIamPolicy` or `containeranalysis.occurrences.setIamPolicy` permission if the resource is a note or occurrence, respectively. The resource takes the format `projects/[PROJECT_ID]/notes/[NOTE_ID]` for notes and `projects/[PROJECT_ID]/occurrences/[OCCURRENCE_ID]` for occurrences.

      Args:
        request: (ContaineranalysisProjectsNotesGetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('GetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    GetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/notes/{notesId}:getIamPolicy', http_method='POST', method_id='containeranalysis.projects.notes.getIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:getIamPolicy', request_field='getIamPolicyRequest', request_type_name='ContaineranalysisProjectsNotesGetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def List(self, request, global_params=None):
        """Lists notes for the specified project.

      Args:
        request: (ContaineranalysisProjectsNotesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListNotesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/notes', http_method='GET', method_id='containeranalysis.projects.notes.list', ordered_params=['parent'], path_params=['parent'], query_params=['filter', 'pageSize', 'pageToken'], relative_path='v1/{+parent}/notes', request_field='', request_type_name='ContaineranalysisProjectsNotesListRequest', response_type_name='ListNotesResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Updates the specified note.

      Args:
        request: (ContaineranalysisProjectsNotesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Note) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/notes/{notesId}', http_method='PATCH', method_id='containeranalysis.projects.notes.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v1/{+name}', request_field='note', request_type_name='ContaineranalysisProjectsNotesPatchRequest', response_type_name='Note', supports_download=False)

    def SetIamPolicy(self, request, global_params=None):
        """Sets the access control policy on the specified note or occurrence. Requires `containeranalysis.notes.setIamPolicy` or `containeranalysis.occurrences.setIamPolicy` permission if the resource is a note or an occurrence, respectively. The resource takes the format `projects/[PROJECT_ID]/notes/[NOTE_ID]` for notes and `projects/[PROJECT_ID]/occurrences/[OCCURRENCE_ID]` for occurrences.

      Args:
        request: (ContaineranalysisProjectsNotesSetIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Policy) The response message.
      """
        config = self.GetMethodConfig('SetIamPolicy')
        return self._RunMethod(config, request, global_params=global_params)
    SetIamPolicy.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/notes/{notesId}:setIamPolicy', http_method='POST', method_id='containeranalysis.projects.notes.setIamPolicy', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:setIamPolicy', request_field='setIamPolicyRequest', request_type_name='ContaineranalysisProjectsNotesSetIamPolicyRequest', response_type_name='Policy', supports_download=False)

    def TestIamPermissions(self, request, global_params=None):
        """Returns the permissions that a caller has on the specified note or occurrence. Requires list permission on the project (for example, `containeranalysis.notes.list`). The resource takes the format `projects/[PROJECT_ID]/notes/[NOTE_ID]` for notes and `projects/[PROJECT_ID]/occurrences/[OCCURRENCE_ID]` for occurrences.

      Args:
        request: (ContaineranalysisProjectsNotesTestIamPermissionsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (TestIamPermissionsResponse) The response message.
      """
        config = self.GetMethodConfig('TestIamPermissions')
        return self._RunMethod(config, request, global_params=global_params)
    TestIamPermissions.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/notes/{notesId}:testIamPermissions', http_method='POST', method_id='containeranalysis.projects.notes.testIamPermissions', ordered_params=['resource'], path_params=['resource'], query_params=[], relative_path='v1/{+resource}:testIamPermissions', request_field='testIamPermissionsRequest', request_type_name='ContaineranalysisProjectsNotesTestIamPermissionsRequest', response_type_name='TestIamPermissionsResponse', supports_download=False)