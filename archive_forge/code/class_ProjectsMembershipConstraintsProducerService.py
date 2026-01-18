from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.anthospolicycontrollerstatus_pa.v1alpha import anthospolicycontrollerstatus_pa_v1alpha_messages as messages
class ProjectsMembershipConstraintsProducerService(base_api.BaseApiService):
    """Service class for the projects_membershipConstraintsProducer resource."""
    _NAME = 'projects_membershipConstraintsProducer'

    def __init__(self, client):
        super(AnthospolicycontrollerstatusPaV1alpha.ProjectsMembershipConstraintsProducerService, self).__init__(client)
        self._upload_configs = {}

    def List(self, request, global_params=None):
        """ListMembershipConstraintsProducer returns per-membership runtime status for constraints. This endpoint is meant for calls originating from Google internal services. The response contains a list of MembershipConstraints. Each MembershipConstraint contains a MembershipRef indicating which member cluster the constraint status corresponds to. Note that if the request is ListMembershipConstraintsProducer(parent=project1) and clusterA is registered to project2 via a Membership in project1, then clusterA's info will appear in the response.

      Args:
        request: (AnthospolicycontrollerstatusPaProjectsMembershipConstraintsProducerListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListMembershipConstraintsProducerResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/membershipConstraintsProducer', http_method='GET', method_id='anthospolicycontrollerstatus_pa.projects.membershipConstraintsProducer.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1alpha/{+parent}/membershipConstraintsProducer', request_field='', request_type_name='AnthospolicycontrollerstatusPaProjectsMembershipConstraintsProducerListRequest', response_type_name='ListMembershipConstraintsProducerResponse', supports_download=False)