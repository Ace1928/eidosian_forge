from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListEffectiveTagsResponse(_messages.Message):
    """The response of ListEffectiveTags.

  Fields:
    effectiveTags: A possibly paginated list of effective tags for the
      specified resource.
    nextPageToken: Pagination token. If the result set is too large to fit in
      a single response, this token is returned. It encodes the position of
      the current result cursor. Feeding this value into a new list request
      with the `page_token` parameter gives the next page of the results. When
      `next_page_token` is not filled in, there is no next page and the list
      returned is the last page in the result set. Pagination tokens have a
      limited lifetime.
  """
    effectiveTags = _messages.MessageField('EffectiveTag', 1, repeated=True)
    nextPageToken = _messages.StringField(2)