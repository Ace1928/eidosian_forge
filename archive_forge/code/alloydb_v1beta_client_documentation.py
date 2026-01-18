from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.alloydb.v1beta import alloydb_v1beta_messages as messages
Lists information about the supported locations for this service.

      Args:
        request: (AlloydbProjectsLocationsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudLocationListLocationsResponse) The response message.
      