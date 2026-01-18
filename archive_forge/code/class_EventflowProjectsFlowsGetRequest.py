from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class EventflowProjectsFlowsGetRequest(_messages.Message):
    """A EventflowProjectsFlowsGetRequest object.

  Fields:
    name: The name of the flow, of the form
      "projects/{projectId}/flows/{flowId}". (Note, this is different from the
      flowId that is stored in flow.metadata.name.)
  """
    name = _messages.StringField(1, required=True)