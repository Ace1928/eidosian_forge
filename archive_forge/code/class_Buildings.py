from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Buildings(_messages.Message):
    """JSON template for Building List Response object in Directory API.

  Fields:
    buildings: The Buildings in this page of results.
    etag: ETag of the resource.
    kind: Kind of resource this is.
    nextPageToken: The continuation token, used to page through large result
      sets. Provide this value in a subsequent request to return the next page
      of results.
  """
    buildings = _messages.MessageField('Building', 1, repeated=True)
    etag = _messages.StringField(2)
    kind = _messages.StringField(3, default=u'admin#directory#resources#buildings#buildingsList')
    nextPageToken = _messages.StringField(4)