from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ServiceconsumermanagementServicesConsumerQuotaMetricsLimitsProducerOverridesPatchRequest(_messages.Message):
    """A ServiceconsumermanagementServicesConsumerQuotaMetricsLimitsProducerOve
  rridesPatchRequest object.

  Fields:
    force: Whether to force the update of the quota override. If updating an
      override would cause the effective quota for the consumer to decrease by
      more than 10 percent, the call is rejected, as a safety measure to avoid
      accidentally decreasing quota too quickly. Setting the force parameter
      to true ignores this restriction.
    name: The resource name of the override to update.  An example name would
      be: `services/compute.googleapis.com/projects/123/consumerQuotaMetrics/c
      ompute.googleapis.com%2Fcpus/limits/%2Fproject%2Fregion/producerOverride
      s/4a3f2c1d`
    updateMask: Update only the specified fields. If unset, all modifiable
      fields will be updated.
    v1Beta1QuotaOverride: A V1Beta1QuotaOverride resource to be passed as the
      request body.
  """
    force = _messages.BooleanField(1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)
    v1Beta1QuotaOverride = _messages.MessageField('V1Beta1QuotaOverride', 4)