from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.securitycenter.v2 import securitycenter_v2_messages as messages
class ProjectsLocationsFindingsService(base_api.BaseApiService):
    """Service class for the projects_locations_findings resource."""
    _NAME = 'projects_locations_findings'

    def __init__(self, client):
        super(SecuritycenterV2.ProjectsLocationsFindingsService, self).__init__(client)
        self._upload_configs = {}

    def BulkMute(self, request, global_params=None):
        """Kicks off an LRO to bulk mute findings for a parent based on a filter. If no location is specified, findings are muted in global. The parent can be either an organization, folder, or project. The findings matched by the filter will be muted after the LRO is done.

      Args:
        request: (SecuritycenterProjectsLocationsFindingsBulkMuteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
        config = self.GetMethodConfig('BulkMute')
        return self._RunMethod(config, request, global_params=global_params)
    BulkMute.method_config = lambda: base_api.ApiMethodInfo(flat_path='v2/projects/{projectsId}/locations/{locationsId}/findings:bulkMute', http_method='POST', method_id='securitycenter.projects.locations.findings.bulkMute', ordered_params=['parent'], path_params=['parent'], query_params=[], relative_path='v2/{+parent}/findings:bulkMute', request_field='bulkMuteFindingsRequest', request_type_name='SecuritycenterProjectsLocationsFindingsBulkMuteRequest', response_type_name='Operation', supports_download=False)