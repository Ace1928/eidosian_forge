from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1ListDataScanJobsResponse(_messages.Message):
    """List DataScanJobs response.

  Fields:
    dataScanJobs: DataScanJobs (BASIC view only) under a given dataScan.
    nextPageToken: Token to retrieve the next page of results, or empty if
      there are no more results in the list.
  """
    dataScanJobs = _messages.MessageField('GoogleCloudDataplexV1DataScanJob', 1, repeated=True)
    nextPageToken = _messages.StringField(2)