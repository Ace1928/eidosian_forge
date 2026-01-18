from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.sasportal.v1alpha1 import sasportal_v1alpha1_messages as messages
def SetupSasAnalytics(self, request, global_params=None):
    """Setups the a GCP Project to receive SAS Analytics messages via GCP Pub/Sub with a subscription to BigQuery. All the Pub/Sub topics and BigQuery tables are created automatically as part of this service.

      Args:
        request: (SasPortalSetupSasAnalyticsRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SasPortalOperation) The response message.
      """
    config = self.GetMethodConfig('SetupSasAnalytics')
    return self._RunMethod(config, request, global_params=global_params)