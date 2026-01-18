from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListInternalRangesResponse(_messages.Message):
    """Response for InternalRange.ListInternalRanges

  Fields:
    internalRanges: Internal ranges to be returned.
    nextPageToken: The next pagination token in the List response. It should
      be used as page_token for the following request. An empty value means no
      more result.
    unreachable: Locations that could not be reached.
  """
    internalRanges = _messages.MessageField('InternalRange', 1, repeated=True)
    nextPageToken = _messages.StringField(2)
    unreachable = _messages.StringField(3, repeated=True)