from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.services import exceptions
from googlecloudsdk.api_lib.util import apis
def ListQuotaMetrics(service, consumer, page_size=None, limit=None):
    """List service quota metrics for a consumer.

  Args:
    service: The service to list metrics for.
    consumer: The consumer to list metrics for, e.g. "projects/123".
    page_size: The page size to list.
    limit: The max number of metrics to return.

  Raises:
    exceptions.PermissionDeniedException: when listing metrics fails.
    apitools_exceptions.HttpError: Another miscellaneous error with the service.

  Returns:
    The list of quota metrics
  """
    _ValidateConsumer(consumer)
    client = _GetClientInstance()
    messages = client.MESSAGES_MODULE
    request = messages.ServiceconsumermanagementServicesConsumerQuotaMetricsListRequest(parent=_SERVICE_CONSUMER_RESOURCE % (service, consumer))
    return list_pager.YieldFromList(client.services_consumerQuotaMetrics, request, limit=limit, batch_size_attribute='pageSize', batch_size=page_size, field='metrics')