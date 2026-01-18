from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SearchBlueprintRevisionsResponse(_messages.Message):
    """Response object for `SearchBlueprintRevisions`.

  Fields:
    blueprints: The list of requested blueprint revisions.
    nextPageToken: A token that can be sent as `page_token` to retrieve the
      next page. If this field is omitted, there are no subsequent pages.
  """
    blueprints = _messages.MessageField('Blueprint', 1, repeated=True)
    nextPageToken = _messages.StringField(2)