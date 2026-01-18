from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class QuotaInfo(_messages.Message):
    """Metadata about an individual quota, containing usage and limit
  information.

  Fields:
    currentUsage: The usage data for this quota as it applies to the current
      limit.
    historicalUsage: The historical usage data of this quota limit. Currently
      it is only available for daily quota limit, that is, base_limit.duration
      = "1d".
    limit: The effective limit for this quota.
  """
    currentUsage = _messages.MessageField('QuotaUsage', 1)
    historicalUsage = _messages.MessageField('QuotaUsage', 2, repeated=True)
    limit = _messages.MessageField('EffectiveQuotaLimit', 3)