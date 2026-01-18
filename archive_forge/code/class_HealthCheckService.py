from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HealthCheckService(_messages.Message):
    """Represents a Health-Check as a Service resource.

  Enums:
    HealthStatusAggregationPolicyValueValuesEnum: Optional. Policy for how the
      results from multiple health checks for the same endpoint are
      aggregated. Defaults to NO_AGGREGATION if unspecified. - NO_AGGREGATION.
      An EndpointHealth message is returned for each pair in the health check
      service. - AND. If any health check of an endpoint reports UNHEALTHY,
      then UNHEALTHY is the HealthState of the endpoint. If all health checks
      report HEALTHY, the HealthState of the endpoint is HEALTHY. . This is
      only allowed with regional HealthCheckService.
    HealthStatusAggregationStrategyValueValuesEnum: This field is deprecated.
      Use health_status_aggregation_policy instead. Policy for how the results
      from multiple health checks for the same endpoint are aggregated. -
      NO_AGGREGATION. An EndpointHealth message is returned for each backend
      in the health check service. - AND. If any backend's health check
      reports UNHEALTHY, then UNHEALTHY is the HealthState of the entire
      health check service. If all backend's are healthy, the HealthState of
      the health check service is HEALTHY. .

  Fields:
    creationTimestamp: [Output Only] Creation timestamp in RFC3339 text
      format.
    description: An optional description of this resource. Provide this
      property when you create the resource.
    fingerprint: Fingerprint of this resource. A hash of the contents stored
      in this object. This field is used in optimistic locking. This field
      will be ignored when inserting a HealthCheckService. An up-to-date
      fingerprint must be provided in order to patch/update the
      HealthCheckService; Otherwise, the request will fail with error 412
      conditionNotMet. To see the latest fingerprint, make a get() request to
      retrieve the HealthCheckService.
    healthChecks: A list of URLs to the HealthCheck resources. Must have at
      least one HealthCheck, and not more than 10 for regional
      HealthCheckService, and not more than 1 for global HealthCheckService.
      HealthCheck resources must have portSpecification=USE_SERVING_PORT or
      portSpecification=USE_FIXED_PORT. For regional HealthCheckService, the
      HealthCheck must be regional and in the same region. For global
      HealthCheckService, HealthCheck must be global. Mix of regional and
      global HealthChecks is not supported. Multiple regional HealthChecks
      must belong to the same region. Regional HealthChecks must belong to the
      same region as zones of NetworkEndpointGroups. For global
      HealthCheckService using global INTERNET_IP_PORT NetworkEndpointGroups,
      the global HealthChecks must specify sourceRegions, and HealthChecks
      that specify sourceRegions can only be used with global INTERNET_IP_PORT
      NetworkEndpointGroups.
    healthStatusAggregationPolicy: Optional. Policy for how the results from
      multiple health checks for the same endpoint are aggregated. Defaults to
      NO_AGGREGATION if unspecified. - NO_AGGREGATION. An EndpointHealth
      message is returned for each pair in the health check service. - AND. If
      any health check of an endpoint reports UNHEALTHY, then UNHEALTHY is the
      HealthState of the endpoint. If all health checks report HEALTHY, the
      HealthState of the endpoint is HEALTHY. . This is only allowed with
      regional HealthCheckService.
    healthStatusAggregationStrategy: This field is deprecated. Use
      health_status_aggregation_policy instead. Policy for how the results
      from multiple health checks for the same endpoint are aggregated. -
      NO_AGGREGATION. An EndpointHealth message is returned for each backend
      in the health check service. - AND. If any backend's health check
      reports UNHEALTHY, then UNHEALTHY is the HealthState of the entire
      health check service. If all backend's are healthy, the HealthState of
      the health check service is HEALTHY. .
    id: [Output Only] The unique identifier for the resource. This identifier
      is defined by the server.
    kind: [Output only] Type of the resource. Always
      compute#healthCheckServicefor health check services.
    name: Name of the resource. The name must be 1-63 characters long, and
      comply with RFC1035. Specifically, the name must be 1-63 characters long
      and match the regular expression `[a-z]([-a-z0-9]*[a-z0-9])?` which
      means the first character must be a lowercase letter, and all following
      characters must be a dash, lowercase letter, or digit, except the last
      character, which cannot be a dash.
    networkEndpointGroups: A list of URLs to the NetworkEndpointGroup
      resources. Must not have more than 100. For regional HealthCheckService,
      NEGs must be in zones in the region of the HealthCheckService. For
      global HealthCheckServices, the NetworkEndpointGroups must be global
      INTERNET_IP_PORT.
    notificationEndpoints: A list of URLs to the NotificationEndpoint
      resources. Must not have more than 10. A list of endpoints for receiving
      notifications of change in health status. For regional
      HealthCheckService, NotificationEndpoint must be regional and in the
      same region. For global HealthCheckService, NotificationEndpoint must be
      global.
    region: [Output Only] URL of the region where the health check service
      resides. This field is not applicable to global health check services.
      You must specify this field as part of the HTTP request URL. It is not
      settable as a field in the request body.
    selfLink: [Output Only] Server-defined URL for the resource.
  """

    class HealthStatusAggregationPolicyValueValuesEnum(_messages.Enum):
        """Optional. Policy for how the results from multiple health checks for
    the same endpoint are aggregated. Defaults to NO_AGGREGATION if
    unspecified. - NO_AGGREGATION. An EndpointHealth message is returned for
    each pair in the health check service. - AND. If any health check of an
    endpoint reports UNHEALTHY, then UNHEALTHY is the HealthState of the
    endpoint. If all health checks report HEALTHY, the HealthState of the
    endpoint is HEALTHY. . This is only allowed with regional
    HealthCheckService.

    Values:
      AND: If any backend's health check reports UNHEALTHY, then UNHEALTHY is
        the HealthState of the entire health check service. If all backend's
        are healthy, the HealthState of the health check service is HEALTHY.
      NO_AGGREGATION: An EndpointHealth message is returned for each backend
        in the health check service.
    """
        AND = 0
        NO_AGGREGATION = 1

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
    creationTimestamp = _messages.StringField(1)
    description = _messages.StringField(2)
    fingerprint = _messages.BytesField(3)
    healthChecks = _messages.StringField(4, repeated=True)
    healthStatusAggregationPolicy = _messages.EnumField('HealthStatusAggregationPolicyValueValuesEnum', 5)
    healthStatusAggregationStrategy = _messages.EnumField('HealthStatusAggregationStrategyValueValuesEnum', 6)
    id = _messages.IntegerField(7, variant=_messages.Variant.UINT64)
    kind = _messages.StringField(8, default='compute#healthCheckService')
    name = _messages.StringField(9)
    networkEndpointGroups = _messages.StringField(10, repeated=True)
    notificationEndpoints = _messages.StringField(11, repeated=True)
    region = _messages.StringField(12)
    selfLink = _messages.StringField(13)