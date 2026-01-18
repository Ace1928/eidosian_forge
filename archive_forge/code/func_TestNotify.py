from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.remotebuildexecution.v1alpha import remotebuildexecution_v1alpha_messages as messages
def TestNotify(self, request, global_params=None):
    """Sends a test notification to the specified instance. Returns a `google.protobuf.Empty` on success.

      Args:
        request: (RemotebuildexecutionProjectsInstancesTestNotifyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleProtobufEmpty) The response message.
      """
    config = self.GetMethodConfig('TestNotify')
    return self._RunMethod(config, request, global_params=global_params)