from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EffectiveQuotaLimit(_messages.Message):
    """An effective quota limit contains the metadata for a quota limit as
  derived from the service config, together with fields that describe the
  effective limit value and what overrides can be applied to it.

  Fields:
    baseLimit: The service's configuration for this quota limit.
    effectiveLimit: The effective limit value, based on the stored producer
      and consumer overrides and the service defaults.
    key: The key used to identify this limit when applying overrides. The
      consumer_overrides and producer_overrides maps are keyed by strings of
      the form "QuotaGroupName/QuotaLimitName".
    maxConsumerOverrideAllowed: The maximum override value that a consumer may
      specify.
  """
    baseLimit = _messages.MessageField('QuotaLimit', 1)
    effectiveLimit = _messages.IntegerField(2)
    key = _messages.StringField(3)
    maxConsumerOverrideAllowed = _messages.IntegerField(4)