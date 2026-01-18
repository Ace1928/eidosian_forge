from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.anthospolicycontrollerstatus_pa.v1alpha import anthospolicycontrollerstatus_pa_v1alpha_messages as messages
class ProjectsMembershipConstraintTemplatesService(base_api.BaseApiService):
    """Service class for the projects_membershipConstraintTemplates resource."""
    _NAME = 'projects_membershipConstraintTemplates'

    def __init__(self, client):
        super(AnthospolicycontrollerstatusPaV1alpha.ProjectsMembershipConstraintTemplatesService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Retrieves status for a single membership constraint template on a single member cluster.

      Args:
        request: (AnthospolicycontrollerstatusPaProjectsMembershipConstraintTemplatesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (MembershipConstraintTemplate) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/membershipConstraintTemplates/{membershipConstraintTemplatesId}/{membershipConstraintTemplatesId1}', http_method='GET', method_id='anthospolicycontrollerstatus_pa.projects.membershipConstraintTemplates.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha/{+name}', request_field='', request_type_name='AnthospolicycontrollerstatusPaProjectsMembershipConstraintTemplatesGetRequest', response_type_name='MembershipConstraintTemplate', supports_download=False)

    def List(self, request, global_params=None):
        """Lists status for constraint templates. Each entry in the response has a ConstraintTemplateRef and MembershipRef, corresponding to status aggregated across all resources within a single member cluster, in pseudocode the response's shape is: [StatusForConstraintTemplate1OnMemberClusterA, StatusForConstraintTemplate2OnMemberClusterA, StatusForConstraintTemplate1OnMemberClusterB, StatusForConstraintTemplate3OnMemberClusterC, ...].

      Args:
        request: (AnthospolicycontrollerstatusPaProjectsMembershipConstraintTemplatesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListMembershipConstraintTemplatesResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/membershipConstraintTemplates', http_method='GET', method_id='anthospolicycontrollerstatus_pa.projects.membershipConstraintTemplates.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1alpha/{+parent}/membershipConstraintTemplates', request_field='', request_type_name='AnthospolicycontrollerstatusPaProjectsMembershipConstraintTemplatesListRequest', response_type_name='ListMembershipConstraintTemplatesResponse', supports_download=False)