from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListExamplesResponse(_messages.Message):
    """Response message for ListExamples.

  Fields:
    examples: The sentence pairs.
    nextPageToken: A token to retrieve next page of results. Pass this token
      to the page_token field in the ListExamplesRequest to obtain the
      corresponding page.
  """
    examples = _messages.MessageField('Example', 1, repeated=True)
    nextPageToken = _messages.StringField(2)