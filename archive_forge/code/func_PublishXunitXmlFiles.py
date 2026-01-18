from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.toolresults.v1beta3 import toolresults_v1beta3_messages as messages
def PublishXunitXmlFiles(self, request, global_params=None):
    """Publish xml files to an existing Step. May return any of the following canonical error codes: - PERMISSION_DENIED - if the user is not authorized to write project - INVALID_ARGUMENT - if the request is malformed - FAILED_PRECONDITION - if the requested state transition is illegal, e.g. try to upload a duplicate xml file or a file too large. - NOT_FOUND - if the containing Execution does not exist.

      Args:
        request: (ToolresultsProjectsHistoriesExecutionsStepsPublishXunitXmlFilesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Step) The response message.
      """
    config = self.GetMethodConfig('PublishXunitXmlFiles')
    return self._RunMethod(config, request, global_params=global_params)