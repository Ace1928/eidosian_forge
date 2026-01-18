from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.apigee.v1 import apigee_v1_messages as messages
def GetResulturl(self, request, global_params=None):
    """After the query is completed, use this API to retrieve the results. If the request succeeds, and there is a non-zero result set, the result is sent to the client as a list of urls to JSON files.

      Args:
        request: (ApigeeOrganizationsEnvironmentsQueriesGetResulturlRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (GoogleCloudApigeeV1GetAsyncQueryResultUrlResponse) The response message.
      """
    config = self.GetMethodConfig('GetResulturl')
    return self._RunMethod(config, request, global_params=global_params)