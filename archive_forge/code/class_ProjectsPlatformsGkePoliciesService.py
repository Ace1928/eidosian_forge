from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.binaryauthorization.v1 import binaryauthorization_v1_messages as messages
class ProjectsPlatformsGkePoliciesService(base_api.BaseApiService):
    """Service class for the projects_platforms_gke_policies resource."""
    _NAME = 'projects_platforms_gke_policies'

    def __init__(self, client):
        super(BinaryauthorizationV1.ProjectsPlatformsGkePoliciesService, self).__init__(client)
        self._upload_configs = {}

    def Evaluate(self, request, global_params=None):
        """Evaluates a Kubernetes object versus a GKE platform policy. Returns `NOT_FOUND` if the policy doesn't exist, `INVALID_ARGUMENT` if the policy or request is malformed and `PERMISSION_DENIED` if the client does not have sufficient permissions.

      Args:
        request: (BinaryauthorizationProjectsPlatformsGkePoliciesEvaluateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (EvaluateGkePolicyResponse) The response message.
      """
        config = self.GetMethodConfig('Evaluate')
        return self._RunMethod(config, request, global_params=global_params)
    Evaluate.method_config = lambda: base_api.ApiMethodInfo(flat_path='v1/projects/{projectsId}/platforms/gke/policies/{policiesId}:evaluate', http_method='POST', method_id='binaryauthorization.projects.platforms.gke.policies.evaluate', ordered_params=['name'], path_params=['name'], query_params=[], relative_path='v1/{+name}:evaluate', request_field='evaluateGkePolicyRequest', request_type_name='BinaryauthorizationProjectsPlatformsGkePoliciesEvaluateRequest', response_type_name='EvaluateGkePolicyResponse', supports_download=False)