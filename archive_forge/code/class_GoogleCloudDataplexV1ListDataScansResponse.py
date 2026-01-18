from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1ListDataScansResponse(_messages.Message):
    """List dataScans response.

  Fields:
    dataScans: DataScans (BASIC view only) under the given parent location.
    nextPageToken: Token to retrieve the next page of results, or empty if
      there are no more results in the list.
    unreachable: Locations that could not be reached.
  """
    dataScans = _messages.MessageField('GoogleCloudDataplexV1DataScan', 1, repeated=True)
    nextPageToken = _messages.StringField(2)
    unreachable = _messages.StringField(3, repeated=True)