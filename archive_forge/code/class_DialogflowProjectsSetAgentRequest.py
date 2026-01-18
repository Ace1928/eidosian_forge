from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DialogflowProjectsSetAgentRequest(_messages.Message):
    """A DialogflowProjectsSetAgentRequest object.

  Fields:
    googleCloudDialogflowV2Agent: A GoogleCloudDialogflowV2Agent resource to
      be passed as the request body.
    parent: Required. The project of this agent. Format: `projects/`.
    updateMask: Optional. The mask to control which fields get updated.
  """
    googleCloudDialogflowV2Agent = _messages.MessageField('GoogleCloudDialogflowV2Agent', 1)
    parent = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)