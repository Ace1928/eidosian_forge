from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowCxV3beta1ExportAgentResponse(_messages.Message):
    """The response message for Agents.ExportAgent.

  Fields:
    agentContent: Uncompressed raw byte content for agent. This field is
      populated if none of `agent_uri` and `git_destination` are specified in
      ExportAgentRequest.
    agentUri: The URI to a file containing the exported agent. This field is
      populated if `agent_uri` is specified in ExportAgentRequest.
    commitSha: Commit SHA of the git push. This field is populated if
      `git_destination` is specified in ExportAgentRequest.
  """
    agentContent = _messages.BytesField(1)
    agentUri = _messages.StringField(2)
    commitSha = _messages.StringField(3)