from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.redis.v1 import redis_v1_messages as messages
def GetAuthString(self, request, global_params=None):
    """Gets the AUTH string for a Redis instance. If AUTH is not enabled for the instance the response will be empty. This information is not included in the details returned to GetInstance.

      Args:
        request: (RedisProjectsLocationsInstancesGetAuthStringRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (InstanceAuthString) The response message.
      """
    config = self.GetMethodConfig('GetAuthString')
    return self._RunMethod(config, request, global_params=global_params)