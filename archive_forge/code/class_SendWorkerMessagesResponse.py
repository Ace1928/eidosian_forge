from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SendWorkerMessagesResponse(_messages.Message):
    """The response to the worker messages.

  Fields:
    workerMessageResponses: The servers response to the worker messages.
  """
    workerMessageResponses = _messages.MessageField('WorkerMessageResponse', 1, repeated=True)