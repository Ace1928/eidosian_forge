from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HttpRouteRouteAction(_messages.Message):
    """The specifications for routing traffic and applying associated policies.

  Fields:
    corsPolicy: The specification for allowing client side cross-origin
      requests.
    destinations: The destination to which traffic should be forwarded.
    directResponse: Optional. Static HTTP Response object to be returned
      regardless of the request.
    faultInjectionPolicy: The specification for fault injection introduced
      into traffic to test the resiliency of clients to backend service
      failure. As part of fault injection, when clients send requests to a
      backend service, delays can be introduced on a percentage of requests
      before sending those requests to the backend service. Similarly requests
      from clients can be aborted for a percentage of requests. timeout and
      retry_policy will be ignored by clients that are configured with a
      fault_injection_policy
    idleTimeout: Optional. Specifies the idle timeout for the selected route.
      The idle timeout is defined as the period in which there are no bytes
      sent or received on either the upstream or downstream connection. If not
      set, the default idle timeout is 1 hour. If set to 0s, the timeout will
      be disabled.
    redirect: If set, the request is directed as configured by this field.
    requestHeaderModifier: The specification for modifying the headers of a
      matching request prior to delivery of the request to the destination. If
      HeaderModifiers are set on both the Destination and the RouteAction,
      they will be merged. Conflicts between the two will not be resolved on
      the configuration.
    requestMirrorPolicy: Specifies the policy on how requests intended for the
      routes destination are shadowed to a separate mirrored destination.
      Proxy will not wait for the shadow destination to respond before
      returning the response. Prior to sending traffic to the shadow service,
      the host/authority header is suffixed with -shadow.
    responseHeaderModifier: The specification for modifying the headers of a
      response prior to sending the response back to the client. If
      HeaderModifiers are set on both the Destination and the RouteAction,
      they will be merged. Conflicts between the two will not be resolved on
      the configuration.
    retryPolicy: Specifies the retry policy associated with this route.
    statefulSessionAffinity: Optional. Specifies cookie-based stateful session
      affinity.
    timeout: Specifies the timeout for selected route. Timeout is computed
      from the time the request has been fully processed (i.e. end of stream)
      up until the response has been completely processed. Timeout includes
      all retries.
    urlRewrite: The specification for rewrite URL before forwarding requests
      to the destination.
  """
    corsPolicy = _messages.MessageField('HttpRouteCorsPolicy', 1)
    destinations = _messages.MessageField('HttpRouteDestination', 2, repeated=True)
    directResponse = _messages.MessageField('HttpRouteHttpDirectResponse', 3)
    faultInjectionPolicy = _messages.MessageField('HttpRouteFaultInjectionPolicy', 4)
    idleTimeout = _messages.StringField(5)
    redirect = _messages.MessageField('HttpRouteRedirect', 6)
    requestHeaderModifier = _messages.MessageField('HttpRouteHeaderModifier', 7)
    requestMirrorPolicy = _messages.MessageField('HttpRouteRequestMirrorPolicy', 8)
    responseHeaderModifier = _messages.MessageField('HttpRouteHeaderModifier', 9)
    retryPolicy = _messages.MessageField('HttpRouteRetryPolicy', 10)
    statefulSessionAffinity = _messages.MessageField('HttpRouteStatefulSessionAffinityPolicy', 11)
    timeout = _messages.StringField(12)
    urlRewrite = _messages.MessageField('HttpRouteURLRewrite', 13)