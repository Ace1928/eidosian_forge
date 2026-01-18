from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.cloudfunctions.v2beta import cloudfunctions_v2beta_messages as messages
Lists information about the supported locations for this service.

      Args:
        request: (CloudfunctionsProjectsLocationsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListLocationsResponse) The response message.
      