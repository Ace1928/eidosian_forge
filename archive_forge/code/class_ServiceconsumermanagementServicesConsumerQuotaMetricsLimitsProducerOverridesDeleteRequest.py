from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ServiceconsumermanagementServicesConsumerQuotaMetricsLimitsProducerOverridesDeleteRequest(_messages.Message):
    """A ServiceconsumermanagementServicesConsumerQuotaMetricsLimitsProducerOve
  rridesDeleteRequest object.

  Fields:
    force: Whether to force the deletion of the quota override. If deleting an
      override would cause the effective quota for the consumer to decrease by
      more than 10 percent, the call is rejected, as a safety measure to avoid
      accidentally decreasing quota too quickly. Setting the force parameter
      to true ignores this restriction.
    name: The resource name of the override to delete.  An example name would
      be: `services/compute.googleapis.com/projects/123/consumerQuotaMetrics/c
      ompute.googleapis.com%2Fcpus/limits/%2Fproject%2Fregion/producerOverride
      s/4a3f2c1d`
  """
    force = _messages.BooleanField(1)
    name = _messages.StringField(2, required=True)