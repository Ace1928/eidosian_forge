from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListImportJobsResponse(_messages.Message):
    """Response message for KeyManagementService.ListImportJobs.

  Fields:
    importJobs: The list of ImportJobs.
    nextPageToken: A token to retrieve next page of results. Pass this value
      in ListImportJobsRequest.page_token to retrieve the next page of
      results.
    totalSize: The total number of ImportJobs that matched the query.
  """
    importJobs = _messages.MessageField('ImportJob', 1, repeated=True)
    nextPageToken = _messages.StringField(2)
    totalSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)