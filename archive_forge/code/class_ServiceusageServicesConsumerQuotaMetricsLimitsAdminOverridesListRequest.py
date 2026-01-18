from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ServiceusageServicesConsumerQuotaMetricsLimitsAdminOverridesListRequest(_messages.Message):
    """A
  ServiceusageServicesConsumerQuotaMetricsLimitsAdminOverridesListRequest
  object.

  Fields:
    pageSize: Requested size of the next page of data.
    pageToken: Token identifying which result to start with; returned by a
      previous list call.
    parent: The resource name of the parent quota limit, returned by a
      ListConsumerQuotaMetrics or GetConsumerQuotaMetric call.  An example
      name would be: `projects/123/services/compute.googleapis.com/consumerQuo
      taMetrics/compute.googleapis.com%2Fcpus/limits/%2Fproject%2Fregion`
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)