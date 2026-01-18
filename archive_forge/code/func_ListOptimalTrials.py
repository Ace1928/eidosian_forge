from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.aiplatform.v1 import aiplatform_v1_messages as messages
def ListOptimalTrials(self, request, global_params=None):
    """Lists the pareto-optimal Trials for multi-objective Study or the optimal Trials for single-objective Study. The definition of pareto-optimal can be checked in wiki page. https://en.wikipedia.org/wiki/Pareto_efficiency.

      Args:
        request: (AiplatformProjectsLocationsStudiesTrialsListOptimalTrialsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1ListOptimalTrialsResponse) The response message.
      """
    config = self.GetMethodConfig('ListOptimalTrials')
    return self._RunMethod(config, request, global_params=global_params)