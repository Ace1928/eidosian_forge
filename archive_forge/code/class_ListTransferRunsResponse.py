from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListTransferRunsResponse(_messages.Message):
    """The returned list of pipelines in the project.

  Fields:
    nextPageToken: Output only. The next-pagination token. For multiple-page
      list results, this token can be used as the
      `ListTransferRunsRequest.page_token` to request the next page of list
      results.
    transferRuns: Output only. The stored pipeline transfer runs.
  """
    nextPageToken = _messages.StringField(1)
    transferRuns = _messages.MessageField('TransferRun', 2, repeated=True)