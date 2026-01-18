from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.anthosevents.v1 import anthosevents_v1_messages as messages
def ReplaceTrigger(self, request, global_params=None):
    """Rpc to replace a trigger. Only the spec and metadata labels and annotations are modifiable. After the Update request, Events for Cloud Run will work to make the 'status' match the requested 'spec'. May provide metadata.resourceVersion to enforce update from last read for optimistic concurrency control.

      Args:
        request: (AnthoseventsNamespacesTriggersReplaceTriggerRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Trigger) The response message.
      """
    config = self.GetMethodConfig('ReplaceTrigger')
    return self._RunMethod(config, request, global_params=global_params)