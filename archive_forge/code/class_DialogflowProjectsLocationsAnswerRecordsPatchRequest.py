from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DialogflowProjectsLocationsAnswerRecordsPatchRequest(_messages.Message):
    """A DialogflowProjectsLocationsAnswerRecordsPatchRequest object.

  Fields:
    googleCloudDialogflowV2AnswerRecord: A GoogleCloudDialogflowV2AnswerRecord
      resource to be passed as the request body.
    name: The unique identifier of this answer record. Format:
      `projects//locations//answerRecords/`.
    updateMask: Required. The mask to control which fields get updated.
  """
    googleCloudDialogflowV2AnswerRecord = _messages.MessageField('GoogleCloudDialogflowV2AnswerRecord', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)