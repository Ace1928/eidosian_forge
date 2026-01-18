from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListRulesResponse(_messages.Message):
    """Response message for RulesService.ListRules.

  Fields:
    nextPageToken: A token, which can be sent as `page_token` to retrieve the
      next page. If this field is omitted, there are no subsequent pages.
    rules: The rules from the specified assetType.
  """
    nextPageToken = _messages.StringField(1)
    rules = _messages.MessageField('Rule', 2, repeated=True)