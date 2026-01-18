from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.policytroubleshooter.v1beta import policytroubleshooter_v1beta_messages as messages
class IamService(base_api.BaseApiService):
    """Service class for the iam resource."""
    _NAME = 'iam'

    def __init__(self, client):
        super(PolicytroubleshooterV1beta.IamService, self).__init__(client)
        self._upload_configs = {}

    def Troubleshoot(self, request, global_params=None):
        """Checks whether a member has a specific permission for a specific resource, and explains why the member does or does not have that permission.

      Args:
        request: (GoogleCloudPolicytroubleshooterV1betaTroubleshootIamPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudPolicytroubleshooterV1betaTroubleshootIamPolicyResponse) The response message.
      """
        config = self.GetMethodConfig('Troubleshoot')
        return self._RunMethod(config, request, global_params=global_params)
    Troubleshoot.method_config = lambda: base_api.ApiMethodInfo(http_method='POST', method_id='policytroubleshooter.iam.troubleshoot', ordered_params=[], path_params=[], query_params=[], relative_path='v1beta/iam:troubleshoot', request_field='<request>', request_type_name='GoogleCloudPolicytroubleshooterV1betaTroubleshootIamPolicyRequest', response_type_name='GoogleCloudPolicytroubleshooterV1betaTroubleshootIamPolicyResponse', supports_download=False)