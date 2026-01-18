from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ServiceusageServicesConsumerQuotaMetricsLimitsConsumerOverridesCreateRequest(_messages.Message):
    """A
  ServiceusageServicesConsumerQuotaMetricsLimitsConsumerOverridesCreateRequest
  object.

  Fields:
    force: Whether to force the creation of the quota override. If creating an
      override would cause the effective quota for the consumer to decrease by
      more than 10 percent, the call is rejected, as a safety measure to avoid
      accidentally decreasing quota too quickly. Setting the force parameter
      to true ignores this restriction.
    parent: The resource name of the parent quota limit, returned by a
      ListConsumerQuotaMetrics or GetConsumerQuotaMetric call.  An example
      name would be: `projects/123/services/compute.googleapis.com/consumerQuo
      taMetrics/compute.googleapis.com%2Fcpus/limits/%2Fproject%2Fregion`
    quotaOverride: A QuotaOverride resource to be passed as the request body.
  """
    force = _messages.BooleanField(1)
    parent = _messages.StringField(2, required=True)
    quotaOverride = _messages.MessageField('QuotaOverride', 3)