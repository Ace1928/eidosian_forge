from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HealthStatusForNetworkEndpoint(_messages.Message):
    """A HealthStatusForNetworkEndpoint object.

  Enums:
    HealthStateValueValuesEnum: Health state of the network endpoint
      determined based on the health checks configured.
    Ipv6HealthStateValueValuesEnum: Health state of the ipv6 network endpoint
      determined based on the health checks configured.

  Fields:
    backendService: URL of the backend service associated with the health
      state of the network endpoint.
    forwardingRule: URL of the forwarding rule associated with the health
      state of the network endpoint.
    healthCheck: URL of the health check associated with the health state of
      the network endpoint.
    healthCheckService: URL of the health check service associated with the
      health state of the network endpoint.
    healthState: Health state of the network endpoint determined based on the
      health checks configured.
    ipv6HealthState: Health state of the ipv6 network endpoint determined
      based on the health checks configured.
  """

    class HealthStateValueValuesEnum(_messages.Enum):
        """Health state of the network endpoint determined based on the health
    checks configured.

    Values:
      DRAINING: Endpoint is being drained.
      HEALTHY: Endpoint is healthy.
      UNHEALTHY: Endpoint is unhealthy.
      UNKNOWN: Health status of the endpoint is unknown.
    """
        DRAINING = 0
        HEALTHY = 1
        UNHEALTHY = 2
        UNKNOWN = 3

    class Ipv6HealthStateValueValuesEnum(_messages.Enum):
        """Health state of the ipv6 network endpoint determined based on the
    health checks configured.

    Values:
      DRAINING: Endpoint is being drained.
      HEALTHY: Endpoint is healthy.
      UNHEALTHY: Endpoint is unhealthy.
      UNKNOWN: Health status of the endpoint is unknown.
    """
        DRAINING = 0
        HEALTHY = 1
        UNHEALTHY = 2
        UNKNOWN = 3
    backendService = _messages.MessageField('BackendServiceReference', 1)
    forwardingRule = _messages.MessageField('ForwardingRuleReference', 2)
    healthCheck = _messages.MessageField('HealthCheckReference', 3)
    healthCheckService = _messages.MessageField('HealthCheckServiceReference', 4)
    healthState = _messages.EnumField('HealthStateValueValuesEnum', 5)
    ipv6HealthState = _messages.EnumField('Ipv6HealthStateValueValuesEnum', 6)