from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudRecaptchaenterpriseV1FraudPreventionAssessmentBehavioralTrustVerdict(_messages.Message):
    """Information about behavioral trust of the transaction.

  Fields:
    trust: Output only. Probability of this transaction attempt being executed
      in a behaviorally trustworthy way. Values are from 0.0 (lowest) to 1.0
      (highest).
  """
    trust = _messages.FloatField(1, variant=_messages.Variant.FLOAT)