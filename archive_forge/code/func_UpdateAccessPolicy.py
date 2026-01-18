from __future__ import absolute_import
from apitools.base.py import base_api
from samples.servicemanagement_sample.servicemanagement_v1 import servicemanagement_v1_messages as messages
from the newest to the oldest.
def UpdateAccessPolicy(self, request, global_params=None):
    """Producer method to update the current policy.  This method will return an.
error if the policy is too large (more than 50 entries across all lists).

      Args:
        request: (ServiceAccessPolicy) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ServiceAccessPolicy) The response message.
      """
    config = self.GetMethodConfig('UpdateAccessPolicy')
    return self._RunMethod(config, request, global_params=global_params)