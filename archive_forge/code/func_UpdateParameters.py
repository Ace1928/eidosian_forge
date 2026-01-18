from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.memcache.v1 import memcache_v1_messages as messages
def UpdateParameters(self, request, global_params=None):
    """Updates the defined Memcached parameters for an existing instance. This method only stages the parameters, it must be followed by `ApplyParameters` to apply the parameters to nodes of the Memcached instance.

      Args:
        request: (MemcacheProjectsLocationsInstancesUpdateParametersRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('UpdateParameters')
    return self._RunMethod(config, request, global_params=global_params)