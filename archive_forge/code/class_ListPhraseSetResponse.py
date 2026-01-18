from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListPhraseSetResponse(_messages.Message):
    """Message returned to the client by the `ListPhraseSet` method.

  Fields:
    nextPageToken: A token, which can be sent as `page_token` to retrieve the
      next page. If this field is omitted, there are no subsequent pages.
    phraseSets: The phrase set.
  """
    nextPageToken = _messages.StringField(1)
    phraseSets = _messages.MessageField('PhraseSet', 2, repeated=True)