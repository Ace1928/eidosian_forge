from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IpAddressSelectionPolicyValueValuesEnum(_messages.Enum):
    """Specifies a preference for traffic sent from the proxy to the backend
    (or from the client to the backend for proxyless gRPC). The possible
    values are: - IPV4_ONLY: Only send IPv4 traffic to the backends of the
    backend service (Instance Group, Managed Instance Group, Network Endpoint
    Group), regardless of traffic from the client to the proxy. Only IPv4
    health checks are used to check the health of the backends. This is the
    default setting. - PREFER_IPV6: Prioritize the connection to the
    endpoint's IPv6 address over its IPv4 address (provided there is a healthy
    IPv6 address). - IPV6_ONLY: Only send IPv6 traffic to the backends of the
    backend service (Instance Group, Managed Instance Group, Network Endpoint
    Group), regardless of traffic from the client to the proxy. Only IPv6
    health checks are used to check the health of the backends. This field is
    applicable to either: - Advanced global external Application Load Balancer
    (load balancing scheme EXTERNAL_MANAGED), - Regional external Application
    Load Balancer, - Internal proxy Network Load Balancer (load balancing
    scheme INTERNAL_MANAGED), - Regional internal Application Load Balancer
    (load balancing scheme INTERNAL_MANAGED), - Traffic Director with Envoy
    proxies and proxyless gRPC (load balancing scheme INTERNAL_SELF_MANAGED).

    Values:
      IPV4_ONLY: Only send IPv4 traffic to the backends of the Backend Service
        (Instance Group, Managed Instance Group, Network Endpoint Group)
        regardless of traffic from the client to the proxy. Only IPv4 health-
        checks are used to check the health of the backends. This is the
        default setting.
      IPV6_ONLY: Only send IPv6 traffic to the backends of the Backend Service
        (Instance Group, Managed Instance Group, Network Endpoint Group)
        regardless of traffic from the client to the proxy. Only IPv6 health-
        checks are used to check the health of the backends.
      IP_ADDRESS_SELECTION_POLICY_UNSPECIFIED: Unspecified IP address
        selection policy.
      PREFER_IPV6: Prioritize the connection to the endpoints IPv6 address
        over its IPv4 address (provided there is a healthy IPv6 address).
    """
    IPV4_ONLY = 0
    IPV6_ONLY = 1
    IP_ADDRESS_SELECTION_POLICY_UNSPECIFIED = 2
    PREFER_IPV6 = 3