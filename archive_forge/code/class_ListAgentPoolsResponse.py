from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListAgentPoolsResponse(_messages.Message):
    """Response from ListAgentPools.

  Fields:
    agentPools: A list of agent pools.
    nextPageToken: The list next page token.
  """
    agentPools = _messages.MessageField('AgentPool', 1, repeated=True)
    nextPageToken = _messages.StringField(2)