from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2ExportAgentRequest(_messages.Message):
    """The request message for Agents.ExportAgent.

  Fields:
    agentUri: Required. The [Google Cloud
      Storage](https://cloud.google.com/storage/docs/) URI to export the agent
      to. The format of this URI must be `gs:///`. If left unspecified, the
      serialized agent is returned inline. Dialogflow performs a write
      operation for the Cloud Storage object on the caller's behalf, so your
      request authentication must have write permissions for the object. For
      more information, see [Dialogflow access
      control](https://cloud.google.com/dialogflow/cx/docs/concept/access-
      control#storage).
  """
    agentUri = _messages.StringField(1)