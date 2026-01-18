from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ConsumerQuotaLimit(_messages.Message):
    """Consumer quota settings for a quota limit.

  Fields:
    allowsAdminOverrides: Whether admin overrides are allowed on this limit
    isPrecise: Whether this limit is precise or imprecise.
    metric: The name of the parent metric of this limit.  An example name
      would be: `compute.googleapis.com/cpus`
    name: The resource name of the quota limit.  An example name would be: `pr
      ojects/123/services/compute.googleapis.com/consumerQuotaMetrics/compute.
      googleapis.com%2Fcpus/limits/%2Fproject%2Fregion`  The resource name is
      intended to be opaque and should not be parsed for its component
      strings, since its representation could change in the future.
    quotaBuckets: Summary of the enforced quota buckets, organized by quota
      dimension, ordered from least specific to most specific (for example,
      the global default bucket, with no quota dimensions, will always appear
      first).
    unit: The limit unit.  An example unit would be `1/{project}/{region}`
      Note that `{project}` and `{region}` are not placeholders in this
      example; the literal characters `{` and `}` occur in the string.
  """
    allowsAdminOverrides = _messages.BooleanField(1)
    isPrecise = _messages.BooleanField(2)
    metric = _messages.StringField(3)
    name = _messages.StringField(4)
    quotaBuckets = _messages.MessageField('QuotaBucket', 5, repeated=True)
    unit = _messages.StringField(6)