from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowCxV3beta1TurnSignals(_messages.Message):
    """Collection of all signals that were extracted for a single turn of the
  conversation.

  Enums:
    FailureReasonsValueListEntryValuesEnum:

  Fields:
    agentEscalated: Whether agent responded with LiveAgentHandoff fulfillment.
    dtmfUsed: Whether user was using DTMF input.
    failureReasons: Failure reasons of the turn.
    noMatch: Whether NLU predicted NO_MATCH.
    noUserInput: Whether user provided no input.
    reachedEndPage: Whether turn resulted in End Session page.
    sentimentMagnitude: Sentiment magnitude of the user utterance if
      [sentiment](https://cloud.google.com/dialogflow/cx/docs/concept/sentimen
      t) was enabled.
    sentimentScore: Sentiment score of the user utterance if
      [sentiment](https://cloud.google.com/dialogflow/cx/docs/concept/sentimen
      t) was enabled.
    userEscalated: Whether user was specifically asking for a live agent.
    webhookStatuses: Human-readable statuses of the webhooks triggered during
      this turn.
  """

    class FailureReasonsValueListEntryValuesEnum(_messages.Enum):
        """FailureReasonsValueListEntryValuesEnum enum type.

    Values:
      FAILURE_REASON_UNSPECIFIED: Failure reason is not assigned.
      FAILED_INTENT: Whether NLU failed to recognize user intent.
      FAILED_WEBHOOK: Whether webhook failed during the turn.
    """
        FAILURE_REASON_UNSPECIFIED = 0
        FAILED_INTENT = 1
        FAILED_WEBHOOK = 2
    agentEscalated = _messages.BooleanField(1)
    dtmfUsed = _messages.BooleanField(2)
    failureReasons = _messages.EnumField('FailureReasonsValueListEntryValuesEnum', 3, repeated=True)
    noMatch = _messages.BooleanField(4)
    noUserInput = _messages.BooleanField(5)
    reachedEndPage = _messages.BooleanField(6)
    sentimentMagnitude = _messages.FloatField(7, variant=_messages.Variant.FLOAT)
    sentimentScore = _messages.FloatField(8, variant=_messages.Variant.FLOAT)
    userEscalated = _messages.BooleanField(9)
    webhookStatuses = _messages.StringField(10, repeated=True)