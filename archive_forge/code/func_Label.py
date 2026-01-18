from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.dataflow.v1b3 import dataflow_v1b3_messages as messages
def Label(self, request, global_params=None):
    """Updates the label of the TemplateVersion. Label can be duplicated in Template, so either add or remove the label in the TemplateVersion.

      Args:
        request: (DataflowProjectsCatalogTemplatesLabelRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ModifyTemplateVersionLabelResponse) The response message.
      """
    config = self.GetMethodConfig('Label')
    return self._RunMethod(config, request, global_params=global_params)