from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.containeranalysis.v1 import containeranalysis_v1_messages as messages
def GetNotes(self, request, global_params=None):
    """Gets the note attached to the specified occurrence. Consumer projects can use this method to get a note that belongs to a provider project.

      Args:
        request: (ContaineranalysisProjectsOccurrencesGetNotesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Note) The response message.
      """
    config = self.GetMethodConfig('GetNotes')
    return self._RunMethod(config, request, global_params=global_params)