from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apigee.v1 import apigee_v1_messages as messages
def ComputeEnvironmentScores(self, request, global_params=None):
    """ComputeEnvironmentScores calculates scores for requested time range for the specified security profile and environment.

      Args:
        request: (ApigeeOrganizationsSecurityProfilesEnvironmentsComputeEnvironmentScoresRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1ComputeEnvironmentScoresResponse) The response message.
      """
    config = self.GetMethodConfig('ComputeEnvironmentScores')
    return self._RunMethod(config, request, global_params=global_params)