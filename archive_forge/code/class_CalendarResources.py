from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CalendarResources(_messages.Message):
    """JSON template for Calendar Resource List Response object in Directory

  API.

  Fields:
    etag: ETag of the resource.
    items: The CalendarResources in this page of results.
    kind: Identifies this as a collection of CalendarResources. This is always
      admin#directory#resources#calendars#calendarResourcesList.
    nextPageToken: The continuation token, used to page through large result
      sets. Provide this value in a subsequent request to return the next page
      of results.
  """
    etag = _messages.StringField(1)
    items = _messages.MessageField('CalendarResource', 2, repeated=True)
    kind = _messages.StringField(3, default=u'admin#directory#resources#calendars#calendarResourcesList')
    nextPageToken = _messages.StringField(4)