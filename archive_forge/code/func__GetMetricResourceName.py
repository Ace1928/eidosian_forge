from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.services import exceptions
from googlecloudsdk.api_lib.util import apis
def _GetMetricResourceName(service, consumer, metric, unit):
    """Get the metric resource name from metric name and unit.

  Args:
    service: The service to manage an override for.
    consumer: The consumer to manage an override for, e.g. "projects/123".
    metric: The quota metric name.
    unit: The unit of quota metric.

  Raises:
    exceptions.Error: when the limit with given metric and unit is not found.

  Returns:
    The quota override operation.
  """
    metrics = ListQuotaMetrics(service, consumer)
    for m in metrics:
        if m.metric == metric:
            for q in m.consumerQuotaLimits:
                if q.unit == unit:
                    return q.name
    raise exceptions.Error('limit not found with name "%s" and unit "%s".' % (metric, unit))