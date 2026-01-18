from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.domains.v1alpha2 import domains_v1alpha2_messages as messages
def RetrieveTransferParameters(self, request, global_params=None):
    """Deprecated: For more information, see [Cloud Domains feature deprecation](https://cloud.google.com/domains/docs/deprecations/feature-deprecations) Gets parameters needed to transfer a domain name from another registrar to Cloud Domains. For domains already managed by [Google Domains](https://domains.google/), use `ImportDomain` instead. Use the returned values to call `TransferDomain`.

      Args:
        request: (DomainsProjectsLocationsRegistrationsRetrieveTransferParametersRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (RetrieveTransferParametersResponse) The response message.
      """
    config = self.GetMethodConfig('RetrieveTransferParameters')
    return self._RunMethod(config, request, global_params=global_params)