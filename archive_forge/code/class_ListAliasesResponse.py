from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ListAliasesResponse(_messages.Message):
    """Response for ListAliases.

  Fields:
    aliases: The list of aliases.
    nextPageToken: Use as the value of page_token in the next call to obtain
      the next page of results. If empty, there are no more results.
    totalAliases: The total number of aliases in the repo of the kind
      specified in the request.
  """
    aliases = _messages.MessageField('Alias', 1, repeated=True)
    nextPageToken = _messages.StringField(2)
    totalAliases = _messages.IntegerField(3, variant=_messages.Variant.INT32)