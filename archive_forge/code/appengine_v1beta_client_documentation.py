from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.appengine.v1beta import appengine_v1beta_messages as messages
Lists information about the supported locations for this service.

      Args:
        request: (AppengineProjectsLocationsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListLocationsResponse) The response message.
      