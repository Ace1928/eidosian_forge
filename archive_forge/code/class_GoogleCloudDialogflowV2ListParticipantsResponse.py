from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2ListParticipantsResponse(_messages.Message):
    """The response message for Participants.ListParticipants.

  Fields:
    nextPageToken: Token to retrieve the next page of results or empty if
      there are no more results in the list.
    participants: The list of participants. There is a maximum number of items
      returned based on the page_size field in the request.
  """
    nextPageToken = _messages.StringField(1)
    participants = _messages.MessageField('GoogleCloudDialogflowV2Participant', 2, repeated=True)