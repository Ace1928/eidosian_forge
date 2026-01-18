from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class V1Beta1ConsumerQuotaLimit(_messages.Message):
    """Consumer quota settings for a quota limit.

  Fields:
    isPrecise: Whether this limit is precise or imprecise.
    metric: The name of the parent metric of this limit.  An example name
      would be: `compute.googleapis.com/cpus`
    name: The resource name of the quota limit.  An example name would be: `se
      rvices/compute.googleapis.com/projects/123/quotas/metrics/compute.google
      apis.com%2Fcpus/limits/%2Fproject%2Fregion`  The resource name is
      intended to be opaque and should not be parsed for its component
      strings, since its representation could change in the future.
    quotaBuckets: Summary of the enforced quota buckets, organized by quota
      dimension, ordered from least specific to most specific (for example,
      the global default bucket, with no quota dimensions, will always appear
      first).
    unit: The limit unit.  An example unit would be: `1/{project}/{region}`
      Note that `{project}` and `{region}` are not placeholders in this
      example; the literal characters `{` and `}` occur in the string.
  """
    isPrecise = _messages.BooleanField(1)
    metric = _messages.StringField(2)
    name = _messages.StringField(3)
    quotaBuckets = _messages.MessageField('V1Beta1QuotaBucket', 4, repeated=True)
    unit = _messages.StringField(5)