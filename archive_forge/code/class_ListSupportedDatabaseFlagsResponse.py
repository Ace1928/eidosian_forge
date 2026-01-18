from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListSupportedDatabaseFlagsResponse(_messages.Message):
    """Message for response to listing SupportedDatabaseFlags.

  Fields:
    nextPageToken: A token identifying a page of results the server should
      return.
    supportedDatabaseFlags: The list of SupportedDatabaseFlags.
  """
    nextPageToken = _messages.StringField(1)
    supportedDatabaseFlags = _messages.MessageField('SupportedDatabaseFlag', 2, repeated=True)