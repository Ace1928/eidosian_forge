from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2SuggestionInput(_messages.Message):
    """Represents the selection of a suggestion.

  Fields:
    answerRecord: Required. The ID of a suggestion selected by the human
      agent. The suggestion(s) were generated in a previous call to request
      Dialogflow assist. The format is: `projects//locations//answerRecords/`
      where is an alphanumeric string.
  """
    answerRecord = _messages.StringField(1)