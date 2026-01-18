from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListTagsResponse(_messages.Message):
    """The response from listing tags.

  Fields:
    nextPageToken: The token to retrieve the next page of tags, or empty if
      there are no more tags to return.
    tags: The tags returned.
  """
    nextPageToken = _messages.StringField(1)
    tags = _messages.MessageField('Tag', 2, repeated=True)