from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ToolresultsProjectsHistoriesExecutionsStepsPatchRequest(_messages.Message):
    """A ToolresultsProjectsHistoriesExecutionsStepsPatchRequest object.

  Fields:
    executionId: A Execution id. Required.
    historyId: A History id. Required.
    projectId: A Project id. Required.
    requestId: A unique request ID for server to detect duplicated requests.
      For example, a UUID. Optional, but strongly recommended.
    step: A Step resource to be passed as the request body.
    stepId: A Step id. Required.
  """
    executionId = _messages.StringField(1, required=True)
    historyId = _messages.StringField(2, required=True)
    projectId = _messages.StringField(3, required=True)
    requestId = _messages.StringField(4)
    step = _messages.MessageField('Step', 5)
    stepId = _messages.StringField(6, required=True)