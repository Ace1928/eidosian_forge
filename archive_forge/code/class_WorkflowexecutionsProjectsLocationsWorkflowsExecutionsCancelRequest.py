from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class WorkflowexecutionsProjectsLocationsWorkflowsExecutionsCancelRequest(_messages.Message):
    """A WorkflowexecutionsProjectsLocationsWorkflowsExecutionsCancelRequest
  object.

  Fields:
    cancelExecutionRequest: A CancelExecutionRequest resource to be passed as
      the request body.
    name: Required. Name of the execution to be cancelled. Format: projects/{p
      roject}/locations/{location}/workflows/{workflow}/executions/{execution}
  """
    cancelExecutionRequest = _messages.MessageField('CancelExecutionRequest', 1)
    name = _messages.StringField(2, required=True)