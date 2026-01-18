from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudRecaptchaenterpriseV1FraudSignals(_messages.Message):
    """Fraud signals describing users and cards involved in the transaction.

  Fields:
    cardSignals: Output only. Signals describing the payment card or cards
      used in this transaction.
    userSignals: Output only. Signals describing the end user in this
      transaction.
  """
    cardSignals = _messages.MessageField('GoogleCloudRecaptchaenterpriseV1FraudSignalsCardSignals', 1)
    userSignals = _messages.MessageField('GoogleCloudRecaptchaenterpriseV1FraudSignalsUserSignals', 2)