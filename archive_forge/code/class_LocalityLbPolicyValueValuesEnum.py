from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LocalityLbPolicyValueValuesEnum(_messages.Enum):
    """The load balancing algorithm used within the scope of the locality.
    The possible values are: - ROUND_ROBIN: This is a simple policy in which
    each healthy backend is selected in round robin order. This is the
    default. - LEAST_REQUEST: An O(1) algorithm which selects two random
    healthy hosts and picks the host which has fewer active requests. -
    RING_HASH: The ring/modulo hash load balancer implements consistent
    hashing to backends. The algorithm has the property that the
    addition/removal of a host from a set of N hosts only affects 1/N of the
    requests. - RANDOM: The load balancer selects a random healthy host. -
    ORIGINAL_DESTINATION: Backend host is selected based on the client
    connection metadata, i.e., connections are opened to the same address as
    the destination address of the incoming connection before the connection
    was redirected to the load balancer. - MAGLEV: used as a drop in
    replacement for the ring hash load balancer. Maglev is not as stable as
    ring hash but has faster table lookup build times and host selection
    times. For more information about Maglev, see
    https://ai.google/research/pubs/pub44824 This field is applicable to
    either: - A regional backend service with the service_protocol set to
    HTTP, HTTPS, or HTTP2, and load_balancing_scheme set to INTERNAL_MANAGED.
    - A global backend service with the load_balancing_scheme set to
    INTERNAL_SELF_MANAGED, INTERNAL_MANAGED, or EXTERNAL_MANAGED. If
    sessionAffinity is not NONE, and this field is not set to MAGLEV or
    RING_HASH, session affinity settings will not take effect. Only
    ROUND_ROBIN and RING_HASH are supported when the backend service is
    referenced by a URL map that is bound to target gRPC proxy that has
    validateForProxyless field set to true.

    Values:
      INVALID_LB_POLICY: <no description>
      LEAST_REQUEST: An O(1) algorithm which selects two random healthy hosts
        and picks the host which has fewer active requests.
      MAGLEV: This algorithm implements consistent hashing to backends. Maglev
        can be used as a drop in replacement for the ring hash load balancer.
        Maglev is not as stable as ring hash but has faster table lookup build
        times and host selection times. For more information about Maglev, see
        https://ai.google/research/pubs/pub44824
      ORIGINAL_DESTINATION: Backend host is selected based on the client
        connection metadata, i.e., connections are opened to the same address
        as the destination address of the incoming connection before the
        connection was redirected to the load balancer.
      RANDOM: The load balancer selects a random healthy host.
      RING_HASH: The ring/modulo hash load balancer implements consistent
        hashing to backends. The algorithm has the property that the
        addition/removal of a host from a set of N hosts only affects 1/N of
        the requests.
      ROUND_ROBIN: This is a simple policy in which each healthy backend is
        selected in round robin order. This is the default.
      WEIGHTED_MAGLEV: Per-instance weighted Load Balancing via health check
        reported weights. If set, the Backend Service must configure a non
        legacy HTTP-based Health Check, and health check replies are expected
        to contain non-standard HTTP response header field X-Load-Balancing-
        Endpoint-Weight to specify the per-instance weights. If set, Load
        Balancing is weighted based on the per-instance weights reported in
        the last processed health check replies, as long as every instance
        either reported a valid weight or had UNAVAILABLE_WEIGHT. Otherwise,
        Load Balancing remains equal-weight. This option is only supported in
        Network Load Balancing.
    """
    INVALID_LB_POLICY = 0
    LEAST_REQUEST = 1
    MAGLEV = 2
    ORIGINAL_DESTINATION = 3
    RANDOM = 4
    RING_HASH = 5
    ROUND_ROBIN = 6
    WEIGHTED_MAGLEV = 7