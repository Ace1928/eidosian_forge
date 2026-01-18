from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.run.v1 import run_v1_messages as messages
class ProjectsAuthorizeddomainsService(base_api.BaseApiService):
    """Service class for the projects_authorizeddomains resource."""
    _NAME = 'projects_authorizeddomains'

    def __init__(self, client):
        super(RunV1.ProjectsAuthorizeddomainsService, self).__init__(client)
        self._upload_configs = {}

    def List(self, request, global_params=None):
        """List authorized domains.

      Args:
        request: (RunProjectsAuthorizeddomainsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListAuthorizedDomainsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/authorizeddomains', http_method='GET', method_id='run.projects.authorizeddomains.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1/{+parent}/authorizeddomains', request_field='', request_type_name='RunProjectsAuthorizeddomainsListRequest', response_type_name='ListAuthorizedDomainsResponse', supports_download=False)