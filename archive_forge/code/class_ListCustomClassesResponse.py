from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListCustomClassesResponse(_messages.Message):
    """Response message for the ListCustomClasses method.

  Fields:
    customClasses: The list of requested CustomClasses.
    nextPageToken: A token, which can be sent as page_token to retrieve the
      next page. If this field is omitted, there are no subsequent pages. This
      token expires after 72 hours.
  """
    customClasses = _messages.MessageField('CustomClass', 1, repeated=True)
    nextPageToken = _messages.StringField(2)