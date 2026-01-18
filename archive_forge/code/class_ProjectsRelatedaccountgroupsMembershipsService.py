from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.recaptchaenterprise.v1 import recaptchaenterprise_v1_messages as messages
class ProjectsRelatedaccountgroupsMembershipsService(base_api.BaseApiService):
    """Service class for the projects_relatedaccountgroups_memberships resource."""
    _NAME = 'projects_relatedaccountgroups_memberships'

    def __init__(self, client):
        super(RecaptchaenterpriseV1.ProjectsRelatedaccountgroupsMembershipsService, self).__init__(client)
        self._upload_configs = {}

    def List(self, request, global_params=None):
        """Get memberships in a group of related accounts.

      Args:
        request: (RecaptchaenterpriseProjectsRelatedaccountgroupsMembershipsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudRecaptchaenterpriseV1ListRelatedAccountGroupMembershipsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/relatedaccountgroups/{relatedaccountgroupsId}/memberships', http_method='GET', method_id='recaptchaenterprise.projects.relatedaccountgroups.memberships.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1/{+parent}/memberships', request_field='', request_type_name='RecaptchaenterpriseProjectsRelatedaccountgroupsMembershipsListRequest', response_type_name='GoogleCloudRecaptchaenterpriseV1ListRelatedAccountGroupMembershipsResponse', supports_download=False)