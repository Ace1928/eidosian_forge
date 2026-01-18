from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.anthospolicycontrollerstatus_pa.v1alpha import anthospolicycontrollerstatus_pa_v1alpha_messages as messages
class ProjectsMembershipConstraintAuditViolationsProducerService(base_api.BaseApiService):
    """Service class for the projects_membershipConstraintAuditViolationsProducer resource."""
    _NAME = 'projects_membershipConstraintAuditViolationsProducer'

    def __init__(self, client):
        super(AnthospolicycontrollerstatusPaV1alpha.ProjectsMembershipConstraintAuditViolationsProducerService, self).__init__(client)
        self._upload_configs = {}

    def List(self, request, global_params=None):
        """ListMembershipConstraintAuditViolationsProducer returns membership specific constraint audit violation info. This endpoint is meant for calls originating from Google internal services.

      Args:
        request: (AnthospolicycontrollerstatusPaProjectsMembershipConstraintAuditViolationsProducerListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListMembershipConstraintAuditViolationsProducerResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/membershipConstraintAuditViolationsProducer', http_method='GET', method_id='anthospolicycontrollerstatus_pa.projects.membershipConstraintAuditViolationsProducer.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1alpha/{+parent}/membershipConstraintAuditViolationsProducer', request_field='', request_type_name='AnthospolicycontrollerstatusPaProjectsMembershipConstraintAuditViolationsProducerListRequest', response_type_name='ListMembershipConstraintAuditViolationsProducerResponse', supports_download=False)