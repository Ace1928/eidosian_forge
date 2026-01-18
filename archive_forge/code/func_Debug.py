from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.appengine.v1beta import appengine_v1beta_messages as messages
def Debug(self, request, global_params=None):
    """Enables debugging on a VM instance. This allows you to use the SSH command to connect to the virtual machine where the instance lives. While in "debug mode", the instance continues to serve live traffic. You should delete the instance when you are done debugging and then allow the system to take over and determine if another instance should be started.Only applicable for instances in App Engine flexible environment.

      Args:
        request: (AppengineAppsServicesVersionsInstancesDebugRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('Debug')
    return self._RunMethod(config, request, global_params=global_params)