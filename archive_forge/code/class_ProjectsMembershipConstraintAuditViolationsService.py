from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.anthospolicycontrollerstatus_pa.v1alpha import anthospolicycontrollerstatus_pa_v1alpha_messages as messages
class ProjectsMembershipConstraintAuditViolationsService(base_api.BaseApiService):
    """Service class for the projects_membershipConstraintAuditViolations resource."""
    _NAME = 'projects_membershipConstraintAuditViolations'

    def __init__(self, client):
        super(AnthospolicycontrollerstatusPaV1alpha.ProjectsMembershipConstraintAuditViolationsService, self).__init__(client)
        self._upload_configs = {}

    def List(self, request, global_params=None):
        """ListMembershipConstraintAuditViolations returns membership specific constraint audit violation info.

      Args:
        request: (AnthospolicycontrollerstatusPaProjectsMembershipConstraintAuditViolationsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListMembershipConstraintAuditViolationsResponse) The response message.
      """
        config = self.GetMethodConfig('List')
        return self._RunMethod(config, request, global_params=global_params)
    List.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1alpha/projects/{projectsId}/membershipConstraintAuditViolations', http_method='GET', method_id='anthospolicycontrollerstatus_pa.projects.membershipConstraintAuditViolations.list', ordered_params=['parent'], path_params=['parent'], query_params=['pageSize', 'pageToken'], relative_path='v1alpha/{+parent}/membershipConstraintAuditViolations', request_field='', request_type_name='AnthospolicycontrollerstatusPaProjectsMembershipConstraintAuditViolationsListRequest', response_type_name='ListMembershipConstraintAuditViolationsResponse', supports_download=False)