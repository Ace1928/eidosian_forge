from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DialogflowProjectsLocationsAgentExportRequest(_messages.Message):
    """A DialogflowProjectsLocationsAgentExportRequest object.

  Fields:
    googleCloudDialogflowV2ExportAgentRequest: A
      GoogleCloudDialogflowV2ExportAgentRequest resource to be passed as the
      request body.
    parent: Required. The project that the agent to export is associated with.
      Format: `projects/`.
  """
    googleCloudDialogflowV2ExportAgentRequest = _messages.MessageField('GoogleCloudDialogflowV2ExportAgentRequest', 1)
    parent = _messages.StringField(2, required=True)