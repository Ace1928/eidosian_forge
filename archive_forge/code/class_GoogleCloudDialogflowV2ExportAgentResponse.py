from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2ExportAgentResponse(_messages.Message):
    """The response message for Agents.ExportAgent.

  Fields:
    agentContent: Zip compressed raw byte content for agent.
    agentUri: The URI to a file containing the exported agent. This field is
      populated only if `agent_uri` is specified in `ExportAgentRequest`.
  """
    agentContent = _messages.BytesField(1)
    agentUri = _messages.StringField(2)