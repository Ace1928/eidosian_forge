from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NonSdkApiInsight(_messages.Message):
    """Non-SDK API insights (to address debugging solutions).

  Fields:
    exampleTraceMessages: Optional sample stack traces, for which this insight
      applies (there should be at least one).
    matcherId: A unique ID, to be used for determining the effectiveness of
      this particular insight in the context of a matcher. (required)
    pendingGoogleUpdateInsight: An insight indicating that the hidden API
      usage originates from a Google-provided library.
    upgradeInsight: An insight indicating that the hidden API usage originates
      from the use of a library that needs to be upgraded.
  """
    exampleTraceMessages = _messages.StringField(1, repeated=True)
    matcherId = _messages.StringField(2)
    pendingGoogleUpdateInsight = _messages.MessageField('PendingGoogleUpdateInsight', 3)
    upgradeInsight = _messages.MessageField('UpgradeInsight', 4)