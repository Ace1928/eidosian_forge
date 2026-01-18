from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.aiplatform.v1 import aiplatform_v1_messages as messages
def ReadBlobData(self, request, global_params=None):
    """Gets bytes of TensorboardBlobs. This is to allow reading blob data stored in consumer project's Cloud Storage bucket without users having to obtain Cloud Storage access permission.

      Args:
        request: (AiplatformProjectsLocationsTensorboardsExperimentsRunsTimeSeriesReadBlobDataRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudAiplatformV1ReadTensorboardBlobDataResponse) The response message.
      """
    config = self.GetMethodConfig('ReadBlobData')
    return self._RunMethod(config, request, global_params=global_params)