from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.services import exceptions
from googlecloudsdk.api_lib.util import apis
def UpdateQuotaOverrideCall(service, consumer, metric, unit, dimensions, value, force=False):
    """Update a quota override.

  Args:
    service: The service to update a quota override for.
    consumer: The consumer to update a quota override for, e.g. "projects/123".
    metric: The quota metric name.
    unit: The unit of quota metric.
    dimensions: The dimensions of the override in dictionary format. It can be
      None.
    value: The override integer value.
    force: Force override update even if the change results in a substantial
      decrease in available quota.

  Raises:
    exceptions.UpdateQuotaOverridePermissionDeniedException: when updating an
    override fails.
    apitools_exceptions.HttpError: Another miscellaneous error with the service.

  Returns:
    The quota override operation.
  """
    _ValidateConsumer(consumer)
    client = _GetClientInstance()
    messages = client.MESSAGES_MODULE
    dimensions_message = _GetDimensions(messages, dimensions)
    request = messages.ServiceconsumermanagementServicesConsumerQuotaMetricsImportProducerOverridesRequest(parent=_SERVICE_CONSUMER_RESOURCE % (service, consumer), v1Beta1ImportProducerOverridesRequest=messages.V1Beta1ImportProducerOverridesRequest(inlineSource=messages.V1Beta1OverrideInlineSource(overrides=[messages.V1Beta1QuotaOverride(metric=metric, unit=unit, overrideValue=value, dimensions=dimensions_message)]), force=force))
    try:
        return client.services_consumerQuotaMetrics.ImportProducerOverrides(request)
    except (apitools_exceptions.HttpForbiddenError, apitools_exceptions.HttpNotFoundError) as e:
        exceptions.ReraiseError(e, exceptions.UpdateQuotaOverridePermissionDeniedException)