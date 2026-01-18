from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudRecaptchaenterpriseV1FraudPreventionAssessment(_messages.Message):
    """Assessment for Fraud Prevention.

  Fields:
    behavioralTrustVerdict: Output only. Assessment of this transaction for
      behavioral trust.
    cardTestingVerdict: Output only. Assessment of this transaction for risk
      of being part of a card testing attack.
    stolenInstrumentVerdict: Output only. Assessment of this transaction for
      risk of a stolen instrument.
    transactionRisk: Output only. Probability of this transaction being
      fraudulent. Summarizes the combined risk of attack vectors below. Values
      are from 0.0 (lowest) to 1.0 (highest).
  """
    behavioralTrustVerdict = _messages.MessageField('GoogleCloudRecaptchaenterpriseV1FraudPreventionAssessmentBehavioralTrustVerdict', 1)
    cardTestingVerdict = _messages.MessageField('GoogleCloudRecaptchaenterpriseV1FraudPreventionAssessmentCardTestingVerdict', 2)
    stolenInstrumentVerdict = _messages.MessageField('GoogleCloudRecaptchaenterpriseV1FraudPreventionAssessmentStolenInstrumentVerdict', 3)
    transactionRisk = _messages.FloatField(4, variant=_messages.Variant.FLOAT)