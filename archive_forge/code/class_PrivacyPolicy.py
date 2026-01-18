from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PrivacyPolicy(_messages.Message):
    """Represents privacy policy that contains the privacy requirements
  specified by the data owner. Currently, this is only supported on views.

  Fields:
    aggregationThresholdPolicy: Optional. Policy used for aggregation
      thresholds.
    differentialPrivacyPolicy: Optional. Policy used for differential privacy.
    joinRestrictionPolicy: Optional. Join restriction policy is outside of the
      one of policies, since this policy can be set along with other policies.
      This policy gives data providers the ability to enforce joins on the
      'join_allowed_columns' when data is queried from a privacy protected
      view.
  """
    aggregationThresholdPolicy = _messages.MessageField('AggregationThresholdPolicy', 1)
    differentialPrivacyPolicy = _messages.MessageField('DifferentialPrivacyPolicy', 2)
    joinRestrictionPolicy = _messages.MessageField('JoinRestrictionPolicy', 3)