from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.aiplatform.v1 import aiplatform_v1_messages as messages
def BatchMigrate(self, request, global_params=None):
    """Batch migrates resources from ml.googleapis.com, automl.googleapis.com, and datalabeling.googleapis.com to Vertex AI.

      Args:
        request: (AiplatformProjectsLocationsMigratableResourcesBatchMigrateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleLongrunningOperation) The response message.
      """
    config = self.GetMethodConfig('BatchMigrate')
    return self._RunMethod(config, request, global_params=global_params)