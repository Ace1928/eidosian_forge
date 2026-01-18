from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListPipelineRunsResponse(_messages.Message):
    """Message for response to listing PipelineRuns

  Fields:
    nextPageToken: A token identifying a page of results the server should
      return.
    pipelineRuns: The list of PipelineRun
  """
    nextPageToken = _messages.StringField(1)
    pipelineRuns = _messages.MessageField('PipelineRun', 2, repeated=True)