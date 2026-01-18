from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatacatalogV1ListTagsResponse(_messages.Message):
    """Response message for ListTags.

  Fields:
    nextPageToken: Pagination token of the next results page. Empty if there
      are no more items in results.
    tags: Tag details.
  """
    nextPageToken = _messages.StringField(1)
    tags = _messages.MessageField('GoogleCloudDatacatalogV1Tag', 2, repeated=True)