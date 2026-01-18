from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GrpcRouteRouteAction(_messages.Message):
    """Specifies how to route matched traffic.

  Fields:
    destinations: Optional. The destination services to which traffic should
      be forwarded. If multiple destinations are specified, traffic will be
      split between Backend Service(s) according to the weight field of these
      destinations.
    faultInjectionPolicy: Optional. The specification for fault injection
      introduced into traffic to test the resiliency of clients to destination
      service failure. As part of fault injection, when clients send requests
      to a destination, delays can be introduced on a percentage of requests
      before sending those requests to the destination service. Similarly
      requests from clients can be aborted by for a percentage of requests.
      timeout and retry_policy will be ignored by clients that are configured
      with a fault_injection_policy
    idleTimeout: Optional. Specifies the idle timeout for the selected route.
      The idle timeout is defined as the period in which there are no bytes
      sent or received on either the upstream or downstream connection. If not
      set, the default idle timeout is 1 hour. If set to 0s, the timeout will
      be disabled.
    retryPolicy: Optional. Specifies the retry policy associated with this
      route.
    statefulSessionAffinity: Optional. Specifies cookie-based stateful session
      affinity.
    timeout: Optional. Specifies the timeout for selected route. Timeout is
      computed from the time the request has been fully processed (i.e. end of
      stream) up until the response has been completely processed. Timeout
      includes all retries.
  """
    destinations = _messages.MessageField('GrpcRouteDestination', 1, repeated=True)
    faultInjectionPolicy = _messages.MessageField('GrpcRouteFaultInjectionPolicy', 2)
    idleTimeout = _messages.StringField(3)
    retryPolicy = _messages.MessageField('GrpcRouteRetryPolicy', 4)
    statefulSessionAffinity = _messages.MessageField('GrpcRouteStatefulSessionAffinityPolicy', 5)
    timeout = _messages.StringField(6)