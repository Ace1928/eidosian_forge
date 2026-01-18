from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataflowProjectsWorkerMessagesRequest(_messages.Message):
    """A DataflowProjectsWorkerMessagesRequest object.

  Fields:
    projectId: The project to send the WorkerMessages to.
    sendWorkerMessagesRequest: A SendWorkerMessagesRequest resource to be
      passed as the request body.
  """
    projectId = _messages.StringField(1, required=True)
    sendWorkerMessagesRequest = _messages.MessageField('SendWorkerMessagesRequest', 2)