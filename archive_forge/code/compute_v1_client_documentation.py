from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.compute.v1 import compute_v1_messages as messages
Retrieves the list of Zone resources available to the specified project.

      Args:
        request: (ComputeZonesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ZoneList) The response message.
      