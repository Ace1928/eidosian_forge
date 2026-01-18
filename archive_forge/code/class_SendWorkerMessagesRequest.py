from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SendWorkerMessagesRequest(_messages.Message):
    """A request for sending worker messages to the service.

  Fields:
    location: The [regional endpoint]
      (https://cloud.google.com/dataflow/docs/concepts/regional-endpoints)
      that contains the job.
    workerMessages: The WorkerMessages to send.
  """
    location = _messages.StringField(1)
    workerMessages = _messages.MessageField('WorkerMessage', 2, repeated=True)