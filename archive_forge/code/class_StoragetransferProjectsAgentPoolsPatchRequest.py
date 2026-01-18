from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StoragetransferProjectsAgentPoolsPatchRequest(_messages.Message):
    """A StoragetransferProjectsAgentPoolsPatchRequest object.

  Fields:
    agentPool: A AgentPool resource to be passed as the request body.
    name: Required. Specifies a unique string that identifies the agent pool.
      Format: `projects/{project_id}/agentPools/{agent_pool_id}`
    updateMask: The [field mask] (https://developers.google.com/protocol-
      buffers/docs/reference/google.protobuf) of the fields in `agentPool` to
      update in this request. The following `agentPool` fields can be updated:
      * display_name * bandwidth_limit
  """
    agentPool = _messages.MessageField('AgentPool', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)