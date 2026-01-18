from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListTaskRunsResponse(_messages.Message):
    """Message for response to listing TaskRuns

  Fields:
    nextPageToken: A token identifying a page of results the server should
      return.
    taskRuns: The list of TaskRun
  """
    nextPageToken = _messages.StringField(1)
    taskRuns = _messages.MessageField('TaskRun', 2, repeated=True)