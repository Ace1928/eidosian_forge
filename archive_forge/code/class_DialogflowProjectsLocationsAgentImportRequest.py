from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DialogflowProjectsLocationsAgentImportRequest(_messages.Message):
    """A DialogflowProjectsLocationsAgentImportRequest object.

  Fields:
    googleCloudDialogflowV2ImportAgentRequest: A
      GoogleCloudDialogflowV2ImportAgentRequest resource to be passed as the
      request body.
    parent: Required. The project that the agent to import is associated with.
      Format: `projects/`.
  """
    googleCloudDialogflowV2ImportAgentRequest = _messages.MessageField('GoogleCloudDialogflowV2ImportAgentRequest', 1)
    parent = _messages.StringField(2, required=True)