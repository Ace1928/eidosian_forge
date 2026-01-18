from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.anthosevents.v1beta1 import anthosevents_v1beta1_messages as messages
Rpc to list triggers in all namespaces.

      Args:
        request: (AnthoseventsTriggersListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ListTriggersResponse) The response message.
      