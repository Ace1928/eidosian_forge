from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.anthospolicycontrollerstatus_pa.v1alpha import anthospolicycontrollerstatus_pa_v1alpha_messages as messages
class ProjectsMembershipConstraintsService(base_api.BaseApiService):
    """Service class for the projects_membershipConstraints resource."""
    _NAME = 'projects_membershipConstraints'

    def __init__(self, client):
        super(AnthospolicycontrollerstatusPaV1alpha.ProjectsMembershipConstraintsService, self).__init__(client)
        self._upload_configs = {}

    def Get(self, request, global_params=None):
        """Retrieves status for a single constraint on a single member cluster.

      Args:
        request: (AnthospolicycontrollerstatusPaProjectsMembershipConstraintsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (MembershipConstraint) The response message.
      """
        config = self.GetMethodConfig('Get')
        return self._RunMethod(config, request, global_params=global_params)
    Get.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/membershipConstraints/{membershipConstraintsId}/{membershipConstraintsId1}/{membershipConstraintsId2}', http_method='GET', method_id='anthospolicycontrollerstatus_pa.projects.membershipConstraints.get', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1alpha/{+name}', request_field='', request_type_name='AnthospolicycontrollerstatusPaProjectsMembershipConstraintsGetRequest', response_type_name='MembershipConstraint', supports_download=False)

    def List(self, request, global_params=None):
        """ListMembershipConstraints returns per-membership runtime status for constraints. The response contains a list of MembershipConstraints. Each MembershipConstraint contains a MembershipRef indicating which member cluster the constraint status corresponds to. Note that if the request is ListMembershipConstraints(parent=project1) and clusterA is registered to project2 via a Membership in project1, then clusterA's info will appear in the response.

      Args:
        request: (AnthospolicycontrollerstatusPaProjectsMembershipConstraintsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListMembershipConstraintsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/membershipConstraints', http_method='GET', method_id='anthospolicycontrollerstatus_pa.projects.membershipConstraints.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1alpha/{+parent}/membershipConstraints', request_field='', request_type_name='AnthospolicycontrollerstatusPaProjectsMembershipConstraintsListRequest', response_type_name='ListMembershipConstraintsResponse', supports_download=False)