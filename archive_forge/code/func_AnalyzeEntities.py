from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.language.v1 import language_v1_messages as messages
def AnalyzeEntities(self, request, global_params=None):
    """Finds named entities (currently proper names and common nouns) in the text along with entity types, salience, mentions for each entity, and other properties.

      Args:
        request: (AnalyzeEntitiesRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (AnalyzeEntitiesResponse) The response message.
      """
    config = self.GetMethodConfig('AnalyzeEntities')
    return self._RunMethod(config, request, global_params=global_params)