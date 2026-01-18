from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EffectiveQuotaLimit2(_messages.Message):
    """An effective quota limit contains the metadata for a quota limit as
  derived from the service config, together with fields that describe the
  effective limit value and what overrides can be applied to it. This is used
  only for quota limits that are grouped by metrics instead of quota groups.

  Fields:
    allowAdminOverrides: whether admin overrides are allowed on this limit.
      Admin overrides are allowed if this limit is an organization level one,
      or if this limit is a project level one and there is an identical
      organizational limit.
    baseLimit: The service's configuration for this quota limit.
    defaultLimit: The default quota limit based on the consumer's reputation
      and billing status. Region and zone default limits are kept.
    quotaBuckets: Effective quota limit, maximum override allowed, and usage
      for each quota bucket.
  """
    allowAdminOverrides = _messages.BooleanField(1)
    baseLimit = _messages.MessageField('QuotaLimit', 2)
    defaultLimit = _messages.MessageField('QuotaLimit', 3)
    quotaBuckets = _messages.MessageField('QuotaBucket', 4, repeated=True)