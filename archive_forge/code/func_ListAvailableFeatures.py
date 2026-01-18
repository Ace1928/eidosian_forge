from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.beta import compute_beta_messages as messages
def ListAvailableFeatures(self, request, global_params=None):
    """Lists all features that can be specified in the SSL policy when using custom profile.

      Args:
        request: (ComputeSslPoliciesListAvailableFeaturesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (SslPoliciesListAvailableFeaturesResponse) The response message.
      """
    config = self.GetMethodConfig('ListAvailableFeatures')
    return self._RunMethod(config, request, global_params=global_params)