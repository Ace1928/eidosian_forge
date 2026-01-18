from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1ListPartitionsResponse(_messages.Message):
    """List metadata partitions response.

  Fields:
    nextPageToken: Token to retrieve the next page of results, or empty if
      there are no remaining results in the list.
    partitions: Partitions under the specified parent entity.
  """
    nextPageToken = _messages.StringField(1)
    partitions = _messages.MessageField('GoogleCloudDataplexV1Partition', 2, repeated=True)