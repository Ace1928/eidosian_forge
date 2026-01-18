from __future__ import absolute_import
from apitools.base.py import base_api
from samples.servicemanagement_sample.servicemanagement_v1 import servicemanagement_v1_messages as messages
from the newest to the oldest.
def GetAccessPolicy(self, request, global_params=None):
    """Producer method to retrieve current policy.

      Args:
        request: (ServicemanagementServicesGetAccessPolicyRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ServiceAccessPolicy) The response message.
      """
    config = self.GetMethodConfig('GetAccessPolicy')
    return self._RunMethod(config, request, global_params=global_params)