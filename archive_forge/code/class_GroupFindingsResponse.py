from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GroupFindingsResponse(_messages.Message):
    """Response message for group by findings.

  Fields:
    groupByResults: Group results. There exists an element for each existing
      unique combination of property/values. The element contains a count for
      the number of times those specific property/values appear.
    nextPageToken: Token to retrieve the next page of results, or empty if
      there are no more results.
    totalSize: The total number of results matching the query.
  """
    groupByResults = _messages.MessageField('GroupResult', 1, repeated=True)
    nextPageToken = _messages.StringField(2)
    totalSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)