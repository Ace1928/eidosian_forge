from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.securitycenter.v2 import securitycenter_v2_messages as messages
class ProjectsSourcesFindingsService(base_api.BaseApiService):
    """Service class for the projects_sources_findings resource."""
    _NAME = 'projects_sources_findings'

    def __init__(self, client):
        super(SecuritycenterV2.ProjectsSourcesFindingsService, self).__init__(client)
        self._upload_configs = {}

    def Group(self, request, global_params=None):
        """Filters an organization or source's findings and groups them by their specified properties in a location. If no location is specified, findings are assumed to be in global To group across all sources provide a `-` as the source id. The following list shows some examples: + `/v2/organizations/{organization_id}/sources/-/findings` + `/v2/organizations/{organization_id}/sources/-/locations/{location_id}/findings` + `/v2/folders/{folder_id}/sources/-/findings` + `/v2/folders/{folder_id}/sources/-/locations/{location_id}/findings` + `/v2/projects/{project_id}/sources/-/findings` + `/v2/projects/{project_id}/sources/-/locations/{location_id}/findings`.

      Args:
        request: (SecuritycenterProjectsSourcesFindingsGroupRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GroupFindingsResponse) The response message.
      """
        config = self.GetMethodConfig('Group')
        return self._RunMethod(config, request, global_params=global_params)
    Group.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/sources/{sourcesId}/findings:group', http_method='POST', method_id='securitycenter.projects.sources.findings.group', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v2/{+parent}/findings:group', request_field='groupFindingsRequest', request_type_name='SecuritycenterProjectsSourcesFindingsGroupRequest', response_type_name='GroupFindingsResponse', supports_download=False)

    def List(self, request, global_params=None):
        """Lists an organization or source's findings. To list across all sources for a given location provide a `-` as the source id. If no location is specified, finding are assumed to be in global. The following list shows some examples: + `/v2/organizations/{organization_id}/sources/-/findings` + `/v2/organizations/{organization_id}/sources/-/locations/{location_id}/findings`.

      Args:
        request: (SecuritycenterProjectsSourcesFindingsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListFindingsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/sources/{sourcesId}/findings', http_method='GET', method_id='securitycenter.projects.sources.findings.list', ordered_params=['parent'], path_params=['parent'], query_params=['fieldMask', 'filter', 'orderBy', 'pageSize', 'pageToken'], relative_path='v2/{+parent}/findings', request_field='', request_type_name='SecuritycenterProjectsSourcesFindingsListRequest', response_type_name='ListFindingsResponse', supports_download=False)

    def Patch(self, request, global_params=None):
        """Creates or updates a finding. If no location is specified, finding is assumed to be in global. The corresponding source must exist for a finding creation to succeed.

      Args:
        request: (SecuritycenterProjectsSourcesFindingsPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudSecuritycenterV2Finding) The response message.
      """
        config = self.GetMethodConfig('Patch')
        return self._RunMethod(config, request, global_params=global_params)
    Patch.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/sources/{sourcesId}/findings/{findingsId}', http_method='PATCH', method_id='securitycenter.projects.sources.findings.patch', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v2/{+name}', request_field='googleCloudSecuritycenterV2Finding', request_type_name='SecuritycenterProjectsSourcesFindingsPatchRequest', response_type_name='GoogleCloudSecuritycenterV2Finding', supports_download=False)

    def SetMute(self, request, global_params=None):
        """Updates the mute state of a finding. If no location is specified, finding is assumed to be in global.

      Args:
        request: (SecuritycenterProjectsSourcesFindingsSetMuteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudSecuritycenterV2Finding) The response message.
      """
        config = self.GetMethodConfig('SetMute')
        return self._RunMethod(config, request, global_params=global_params)
    SetMute.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/sources/{sourcesId}/findings/{findingsId}:setMute', http_method='POST', method_id='securitycenter.projects.sources.findings.setMute', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}:setMute', request_field='setMuteRequest', request_type_name='SecuritycenterProjectsSourcesFindingsSetMuteRequest', response_type_name='GoogleCloudSecuritycenterV2Finding', supports_download=False)

    def SetState(self, request, global_params=None):
        """Updates the state of a finding. If no location is specified, finding is assumed to be in global.

      Args:
        request: (SecuritycenterProjectsSourcesFindingsSetStateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudSecuritycenterV2Finding) The response message.
      """
        config = self.GetMethodConfig('SetState')
        return self._RunMethod(config, request, global_params=global_params)
    SetState.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/sources/{sourcesId}/findings/{findingsId}:setState', http_method='POST', method_id='securitycenter.projects.sources.findings.setState', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v2/{+name}:setState', request_field='setFindingStateRequest', request_type_name='SecuritycenterProjectsSourcesFindingsSetStateRequest', response_type_name='GoogleCloudSecuritycenterV2Finding', supports_download=False)

    def UpdateSecurityMarks(self, request, global_params=None):
        """Updates security marks. For Finding Security marks, if no location is specified, finding is assumed to be in global. Assets Security Marks can only be accessed through global endpoint.

      Args:
        request: (SecuritycenterProjectsSourcesFindingsUpdateSecurityMarksRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudSecuritycenterV2SecurityMarks) The response message.
      """
        config = self.GetMethodConfig('UpdateSecurityMarks')
        return self._RunMethod(config, request, global_params=global_params)
    UpdateSecurityMarks.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/sources/{sourcesId}/findings/{findingsId}/securityMarks', http_method='PATCH', method_id='securitycenter.projects.sources.findings.updateSecurityMarks', ordered_params=['name'], path_params=['name'], query_params=['updateMask'], relative_path='v2/{+name}', request_field='googleCloudSecuritycenterV2SecurityMarks', request_type_name='SecuritycenterProjectsSourcesFindingsUpdateSecurityMarksRequest', response_type_name='GoogleCloudSecuritycenterV2SecurityMarks', supports_download=False)