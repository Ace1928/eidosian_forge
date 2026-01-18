from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataflowProjectsLocationsWorkerMessagesRequest(_messages.Message):
    """A DataflowProjectsLocationsWorkerMessagesRequest object.

  Fields:
    location: The [regional endpoint]
      (https://cloud.google.com/dataflow/docs/concepts/regional-endpoints)
      that contains the job.
    projectId: The project to send the WorkerMessages to.
    sendWorkerMessagesRequest: A SendWorkerMessagesRequest resource to be
      passed as the request body.
  """
    location = _messages.StringField(1, required=True)
    projectId = _messages.StringField(2, required=True)
    sendWorkerMessagesRequest = _messages.MessageField('SendWorkerMessagesRequest', 3)