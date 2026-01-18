from __future__ import absolute_import
from apitools.base.py import base_api
from samples.servicemanagement_sample.servicemanagement_v1 import servicemanagement_v1_messages as messages
from the newest to the oldest.
def UpdateConfig(self, request, global_params=None):
    """Updates the specified subset of the service resource. Equivalent to.
calling `UpdateService` with only the `service_config` field updated.

Operation<response: google.api.Service>

      Args:
        request: (ServicemanagementServicesUpdateConfigRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('UpdateConfig')
    return self._RunMethod(config, request, global_params=global_params)