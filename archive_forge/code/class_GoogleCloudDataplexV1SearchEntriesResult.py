from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1SearchEntriesResult(_messages.Message):
    """A single result of a SearchEntries request.

  Fields:
    dataplexEntry: Entry format of the result.
    linkedResource: Linked resource name.
    snippets: Snippets.
  """
    dataplexEntry = _messages.MessageField('GoogleCloudDataplexV1Entry', 1)
    linkedResource = _messages.StringField(2)
    snippets = _messages.MessageField('GoogleCloudDataplexV1SearchEntriesResultSnippets', 3)