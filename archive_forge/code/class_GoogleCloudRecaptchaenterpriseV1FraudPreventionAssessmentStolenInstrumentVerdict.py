from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudRecaptchaenterpriseV1FraudPreventionAssessmentStolenInstrumentVerdict(_messages.Message):
    """Information about stolen instrument fraud, where the user is not the
  legitimate owner of the instrument being used for the purchase.

  Fields:
    risk: Output only. Probability of this transaction being executed with a
      stolen instrument. Values are from 0.0 (lowest) to 1.0 (highest).
  """
    risk = _messages.FloatField(1, variant=_messages.Variant.FLOAT)