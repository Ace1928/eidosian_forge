from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListMessagesResponse(_messages.Message):
    """Lists the messages in the specified HL7v2 store.

  Fields:
    hl7V2Messages: The returned Messages. Won't be more Messages than the
      value of page_size in the request. See view for populated fields.
    nextPageToken: Token to retrieve the next page of results or empty if
      there are no more results in the list.
  """
    hl7V2Messages = _messages.MessageField('Message', 1, repeated=True)
    nextPageToken = _messages.StringField(2)