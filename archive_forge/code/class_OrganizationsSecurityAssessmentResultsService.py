from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apigee.v1 import apigee_v1_messages as messages
class OrganizationsSecurityAssessmentResultsService(base_api.BaseApiService):
    """Service class for the organizations_securityAssessmentResults resource."""
    _NAME = 'organizations_securityAssessmentResults'

    def __init__(self, client):
        super(ApigeeV1.OrganizationsSecurityAssessmentResultsService, self).__init__(client)
        self._upload_configs = {}

    def BatchCompute(self, request, global_params=None):
        """Compute RAV2 security scores for a set of resources.

      Args:
        request: (ApigeeOrganizationsSecurityAssessmentResultsBatchComputeRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1BatchComputeSecurityAssessmentResultsResponse) The response message.
      """
        config = self.GetMethodConfig('BatchCompute')
        return self._RunMethod(config, request, global_params=global_params)
    BatchCompute.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/organizations/{organizationsId}/securityAssessmentResults:batchCompute', http_method='POST', method_id='apigee.organizations.securityAssessmentResults.batchCompute', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:batchCompute', request_field='googleCloudApigeeV1BatchComputeSecurityAssessmentResultsRequest', request_type_name='ApigeeOrganizationsSecurityAssessmentResultsBatchComputeRequest', response_type_name='GoogleCloudApigeeV1BatchComputeSecurityAssessmentResultsResponse', supports_download=False)