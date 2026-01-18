from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HealthStatusAggregationStrategyValueValuesEnum(_messages.Enum):
    """This field is deprecated. Use health_status_aggregation_policy
    instead. Policy for how the results from multiple health checks for the
    same endpoint are aggregated. - NO_AGGREGATION. An EndpointHealth message
    is returned for each backend in the health check service. - AND. If any
    backend's health check reports UNHEALTHY, then UNHEALTHY is the
    HealthState of the entire health check service. If all backend's are
    healthy, the HealthState of the health check service is HEALTHY. .

    Values:
      AND: This is deprecated. Use health_status_aggregation_policy instead.
        If any backend's health check reports UNHEALTHY, then UNHEALTHY is the
        HealthState of the entire health check service. If all backend's are
        healthy, the HealthState of the health check service is HEALTHY.
      NO_AGGREGATION: This is deprecated. Use health_status_aggregation_policy
        instead. An EndpointHealth message is returned for each backend in the
        health check service.
    """
    AND = 0
    NO_AGGREGATION = 1