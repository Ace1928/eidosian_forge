from __future__ import absolute_import
from apitools.base.py import base_api
from googlecloudsdk.generated_clients.apis.dataflow.v1b3 import dataflow_v1b3_messages as messages
def Tag(self, request, global_params=None):
    """Updates the tag of the TemplateVersion, and tag is unique in Template. If tag exists in another TemplateVersion in the Template, updates the tag to this TemplateVersion will remove it from the old TemplateVersion and add it to this TemplateVersion. If request is remove_only (remove_only = true), remove the tag from this TemplateVersion.

      Args:
        request: (DataflowProjectsCatalogTemplatesTagRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ModifyTemplateVersionTagResponse) The response message.
      """
    config = self.GetMethodConfig('Tag')
    return self._RunMethod(config, request, global_params=global_params)