from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudquotas.v1 import cloudquotas_v1_messages as messages
Lists QuotaInfos of all quotas for a given project, folder or organization.

      Args:
        request: (CloudquotasProjectsLocationsServicesQuotaInfosListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListQuotaInfosResponse) The response message.
      