from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.dataplex.v1 import dataplex_v1_messages as messages
def GenerateDataQualityRules(self, request, global_params=None):
    """Generates recommended DataQualityRule from a data profiling DataScan.

      Args:
        request: (DataplexProjectsLocationsDataScansGenerateDataQualityRulesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudDataplexV1GenerateDataQualityRulesResponse) The response message.
      """
    config = self.GetMethodConfig('GenerateDataQualityRules')
    return self._RunMethod(config, request, global_params=global_params)