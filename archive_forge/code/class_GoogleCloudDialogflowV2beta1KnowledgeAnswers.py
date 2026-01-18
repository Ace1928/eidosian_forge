from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2beta1KnowledgeAnswers(_messages.Message):
    """Represents the result of querying a Knowledge base.

  Fields:
    answers: A list of answers from Knowledge Connector.
  """
    answers = _messages.MessageField('GoogleCloudDialogflowV2beta1KnowledgeAnswersAnswer', 1, repeated=True)