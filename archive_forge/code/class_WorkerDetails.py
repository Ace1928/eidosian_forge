from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class WorkerDetails(_messages.Message):
    """Information about a worker

  Fields:
    workItems: Work items processed by this worker, sorted by time.
    workerName: Name of this worker
  """
    workItems = _messages.MessageField('WorkItemDetails', 1, repeated=True)
    workerName = _messages.StringField(2)