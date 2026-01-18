from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.logging.v2 import logging_v2_messages as messages
def UpdateAsync(self, request, global_params=None):
    """Updates a log bucket asynchronously.If the bucket has a lifecycle_state of DELETE_REQUESTED, then FAILED_PRECONDITION will be returned.After a bucket has been created, the bucket's location cannot be changed.

      Args:
        request: (LoggingProjectsLocationsBucketsUpdateAsyncRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
    config = self.GetMethodConfig('UpdateAsync')
    return self._RunMethod(config, request, global_params=global_params)