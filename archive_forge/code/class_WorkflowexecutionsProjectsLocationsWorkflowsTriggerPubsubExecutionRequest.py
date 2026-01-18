from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class WorkflowexecutionsProjectsLocationsWorkflowsTriggerPubsubExecutionRequest(_messages.Message):
    """A
  WorkflowexecutionsProjectsLocationsWorkflowsTriggerPubsubExecutionRequest
  object.

  Fields:
    triggerPubsubExecutionRequest: A TriggerPubsubExecutionRequest resource to
      be passed as the request body.
    workflow: Required. Name of the workflow for which an execution should be
      created. Format:
      projects/{project}/locations/{location}/workflows/{workflow}
  """
    triggerPubsubExecutionRequest = _messages.MessageField('TriggerPubsubExecutionRequest', 1)
    workflow = _messages.StringField(2, required=True)