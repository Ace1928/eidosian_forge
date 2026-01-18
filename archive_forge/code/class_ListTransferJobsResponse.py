from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListTransferJobsResponse(_messages.Message):
    """Response from ListTransferJobs.

  Fields:
    nextPageToken: The list next page token.
    transferJobs: A list of transfer jobs.
  """
    nextPageToken = _messages.StringField(1)
    transferJobs = _messages.MessageField('TransferJob', 2, repeated=True)