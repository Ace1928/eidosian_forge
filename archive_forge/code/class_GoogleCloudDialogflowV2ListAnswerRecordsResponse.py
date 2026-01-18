from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2ListAnswerRecordsResponse(_messages.Message):
    """Response message for AnswerRecords.ListAnswerRecords.

  Fields:
    answerRecords: The list of answer records.
    nextPageToken: A token to retrieve next page of results. Or empty if there
      are no more results. Pass this value in the
      ListAnswerRecordsRequest.page_token field in the subsequent call to
      `ListAnswerRecords` method to retrieve the next page of results.
  """
    answerRecords = _messages.MessageField('GoogleCloudDialogflowV2AnswerRecord', 1, repeated=True)
    nextPageToken = _messages.StringField(2)