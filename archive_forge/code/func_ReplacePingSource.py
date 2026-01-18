from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.anthosevents.v1beta1 import anthosevents_v1beta1_messages as messages
def ReplacePingSource(self, request, global_params=None):
    """Rpc to replace a pingsource. Only the spec and metadata labels and annotations are modifiable. After the Update request, Cloud Run will work to make the 'status' match the requested 'spec'. May provide metadata.resourceVersion to enforce update from last read for optimistic concurrency control.

      Args:
        request: (AnthoseventsNamespacesPingsourcesReplacePingSourceRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (PingSource) The response message.
      """
    config = self.GetMethodConfig('ReplacePingSource')
    return self._RunMethod(config, request, global_params=global_params)