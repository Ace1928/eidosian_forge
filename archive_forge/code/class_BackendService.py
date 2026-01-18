from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BackendService(_messages.Message):
    """Represents a Backend Service resource. A backend service defines how
  Google Cloud load balancers distribute traffic. The backend service
  configuration contains a set of values, such as the protocol used to connect
  to backends, various distribution and session settings, health checks, and
  timeouts. These settings provide fine-grained control over how your load
  balancer behaves. Most of the settings have default values that allow for
  easy configuration if you need to get started quickly. Backend services in
  Google Compute Engine can be either regionally or globally scoped. * [Global
  ](https://cloud.google.com/compute/docs/reference/rest/beta/backendServices)
  * [Regional](https://cloud.google.com/compute/docs/reference/rest/beta/regio
  nBackendServices) For more information, see Backend Services.

  Enums:
    CompressionModeValueValuesEnum: Compress text responses using Brotli or
      gzip compression, based on the client's Accept-Encoding header.
    IpAddressSelectionPolicyValueValuesEnum: Specifies a preference for
      traffic sent from the proxy to the backend (or from the client to the
      backend for proxyless gRPC). The possible values are: - IPV4_ONLY: Only
      send IPv4 traffic to the backends of the backend service (Instance
      Group, Managed Instance Group, Network Endpoint Group), regardless of
      traffic from the client to the proxy. Only IPv4 health checks are used
      to check the health of the backends. This is the default setting. -
      PREFER_IPV6: Prioritize the connection to the endpoint's IPv6 address
      over its IPv4 address (provided there is a healthy IPv6 address). -
      IPV6_ONLY: Only send IPv6 traffic to the backends of the backend service
      (Instance Group, Managed Instance Group, Network Endpoint Group),
      regardless of traffic from the client to the proxy. Only IPv6 health
      checks are used to check the health of the backends. This field is
      applicable to either: - Advanced global external Application Load
      Balancer (load balancing scheme EXTERNAL_MANAGED), - Regional external
      Application Load Balancer, - Internal proxy Network Load Balancer (load
      balancing scheme INTERNAL_MANAGED), - Regional internal Application Load
      Balancer (load balancing scheme INTERNAL_MANAGED), - Traffic Director
      with Envoy proxies and proxyless gRPC (load balancing scheme
      INTERNAL_SELF_MANAGED).
    LoadBalancingSchemeValueValuesEnum: Specifies the load balancer type. A
      backend service created for one type of load balancer cannot be used
      with another. For more information, refer to Choosing a load balancer.
    LocalityLbPolicyValueValuesEnum: The load balancing algorithm used within
      the scope of the locality. The possible values are: - ROUND_ROBIN: This
      is a simple policy in which each healthy backend is selected in round
      robin order. This is the default. - LEAST_REQUEST: An O(1) algorithm
      which selects two random healthy hosts and picks the host which has
      fewer active requests. - RING_HASH: The ring/modulo hash load balancer
      implements consistent hashing to backends. The algorithm has the
      property that the addition/removal of a host from a set of N hosts only
      affects 1/N of the requests. - RANDOM: The load balancer selects a
      random healthy host. - ORIGINAL_DESTINATION: Backend host is selected
      based on the client connection metadata, i.e., connections are opened to
      the same address as the destination address of the incoming connection
      before the connection was redirected to the load balancer. - MAGLEV:
      used as a drop in replacement for the ring hash load balancer. Maglev is
      not as stable as ring hash but has faster table lookup build times and
      host selection times. For more information about Maglev, see
      https://ai.google/research/pubs/pub44824 This field is applicable to
      either: - A regional backend service with the service_protocol set to
      HTTP, HTTPS, or HTTP2, and load_balancing_scheme set to
      INTERNAL_MANAGED. - A global backend service with the
      load_balancing_scheme set to INTERNAL_SELF_MANAGED, INTERNAL_MANAGED, or
      EXTERNAL_MANAGED. If sessionAffinity is not NONE, and this field is not
      set to MAGLEV or RING_HASH, session affinity settings will not take
      effect. Only ROUND_ROBIN and RING_HASH are supported when the backend
      service is referenced by a URL map that is bound to target gRPC proxy
      that has validateForProxyless field set to true.
    ProtocolValueValuesEnum: The protocol this BackendService uses to
      communicate with backends. Possible values are HTTP, HTTPS, HTTP2, TCP,
      SSL, UDP or GRPC. depending on the chosen load balancer or Traffic
      Director configuration. Refer to the documentation for the load
      balancers or for Traffic Director for more information. Must be set to
      GRPC when the backend service is referenced by a URL map that is bound
      to target gRPC proxy.
    SessionAffinityValueValuesEnum: Type of session affinity to use. The
      default is NONE. Only NONE and HEADER_FIELD are supported when the
      backend service is referenced by a URL map that is bound to target gRPC
      proxy that has validateForProxyless field set to true. For more details,
      see: [Session Affinity](https://cloud.google.com/load-
      balancing/docs/backend-service#session_affinity).

  Messages:
    MetadatasValue: Deployment metadata associated with the resource to be set
      by a GKE hub controller and read by the backend RCTH

  Fields:
    affinityCookieTtlSec: Lifetime of cookies in seconds. This setting is
      applicable to Application Load Balancers and Traffic Director and
      requires GENERATED_COOKIE or HTTP_COOKIE session affinity. If set to 0,
      the cookie is non-persistent and lasts only until the end of the browser
      session (or equivalent). The maximum allowed value is two weeks
      (1,209,600). Not supported when the backend service is referenced by a
      URL map that is bound to target gRPC proxy that has validateForProxyless
      field set to true.
    backends: The list of backends that serve this BackendService.
    cdnPolicy: Cloud CDN configuration for this BackendService. Only available
      for specified load balancer types.
    circuitBreakers: A CircuitBreakers attribute.
    compressionMode: Compress text responses using Brotli or gzip compression,
      based on the client's Accept-Encoding header.
    connectionDraining: A ConnectionDraining attribute.
    connectionTrackingPolicy: Connection Tracking configuration for this
      BackendService. Connection tracking policy settings are only available
      for external passthrough Network Load Balancers and internal passthrough
      Network Load Balancers.
    consistentHash: Consistent Hash-based load balancing can be used to
      provide soft session affinity based on HTTP headers, cookies or other
      properties. This load balancing policy is applicable only for HTTP
      connections. The affinity to a particular destination host will be lost
      when one or more hosts are added/removed from the destination service.
      This field specifies parameters that control consistent hashing. This
      field is only applicable when localityLbPolicy is set to MAGLEV or
      RING_HASH. This field is applicable to either: - A regional backend
      service with the service_protocol set to HTTP, HTTPS, or HTTP2, and
      load_balancing_scheme set to INTERNAL_MANAGED. - A global backend
      service with the load_balancing_scheme set to INTERNAL_SELF_MANAGED.
    creationTimestamp: [Output Only] Creation timestamp in RFC3339 text
      format.
    customRequestHeaders: Headers that the load balancer adds to proxied
      requests. See [Creating custom headers](https://cloud.google.com/load-
      balancing/docs/custom-headers).
    customResponseHeaders: Headers that the load balancer adds to proxied
      responses. See [Creating custom headers](https://cloud.google.com/load-
      balancing/docs/custom-headers).
    description: An optional description of this resource. Provide this
      property when you create the resource.
    edgeSecurityPolicy: [Output Only] The resource URL for the edge security
      policy associated with this backend service.
    enableCDN: If true, enables Cloud CDN for the backend service of a global
      external Application Load Balancer.
    failoverPolicy: Requires at least one backend instance group to be defined
      as a backup (failover) backend. For load balancers that have
      configurable failover: [Internal passthrough Network Load
      Balancers](https://cloud.google.com/load-
      balancing/docs/internal/failover-overview) and [external passthrough
      Network Load Balancers](https://cloud.google.com/load-
      balancing/docs/network/networklb-failover-overview).
    fingerprint: Fingerprint of this resource. A hash of the contents stored
      in this object. This field is used in optimistic locking. This field
      will be ignored when inserting a BackendService. An up-to-date
      fingerprint must be provided in order to update the BackendService,
      otherwise the request will fail with error 412 conditionNotMet. To see
      the latest fingerprint, make a get() request to retrieve a
      BackendService.
    healthChecks: The list of URLs to the healthChecks, httpHealthChecks
      (legacy), or httpsHealthChecks (legacy) resource for health checking
      this backend service. Not all backend services support legacy health
      checks. See Load balancer guide. Currently, at most one health check can
      be specified for each backend service. Backend services with instance
      group or zonal NEG backends must have a health check. Backend services
      with internet or serverless NEG backends must not have a health check.
    iap: The configurations for Identity-Aware Proxy on this resource. Not
      available for internal passthrough Network Load Balancers and external
      passthrough Network Load Balancers.
    id: [Output Only] The unique identifier for the resource. This identifier
      is defined by the server.
    ipAddressSelectionPolicy: Specifies a preference for traffic sent from the
      proxy to the backend (or from the client to the backend for proxyless
      gRPC). The possible values are: - IPV4_ONLY: Only send IPv4 traffic to
      the backends of the backend service (Instance Group, Managed Instance
      Group, Network Endpoint Group), regardless of traffic from the client to
      the proxy. Only IPv4 health checks are used to check the health of the
      backends. This is the default setting. - PREFER_IPV6: Prioritize the
      connection to the endpoint's IPv6 address over its IPv4 address
      (provided there is a healthy IPv6 address). - IPV6_ONLY: Only send IPv6
      traffic to the backends of the backend service (Instance Group, Managed
      Instance Group, Network Endpoint Group), regardless of traffic from the
      client to the proxy. Only IPv6 health checks are used to check the
      health of the backends. This field is applicable to either: - Advanced
      global external Application Load Balancer (load balancing scheme
      EXTERNAL_MANAGED), - Regional external Application Load Balancer, -
      Internal proxy Network Load Balancer (load balancing scheme
      INTERNAL_MANAGED), - Regional internal Application Load Balancer (load
      balancing scheme INTERNAL_MANAGED), - Traffic Director with Envoy
      proxies and proxyless gRPC (load balancing scheme
      INTERNAL_SELF_MANAGED).
    kind: [Output Only] Type of resource. Always compute#backendService for
      backend services.
    loadBalancingScheme: Specifies the load balancer type. A backend service
      created for one type of load balancer cannot be used with another. For
      more information, refer to Choosing a load balancer.
    localityLbPolicies: A list of locality load-balancing policies to be used
      in order of preference. When you use localityLbPolicies, you must set at
      least one value for either the localityLbPolicies[].policy or the
      localityLbPolicies[].customPolicy field. localityLbPolicies overrides
      any value set in the localityLbPolicy field. For an example of how to
      use this field, see Define a list of preferred policies. Caution: This
      field and its children are intended for use in a service mesh that
      includes gRPC clients only. Envoy proxies can't use backend services
      that have this configuration.
    localityLbPolicy: The load balancing algorithm used within the scope of
      the locality. The possible values are: - ROUND_ROBIN: This is a simple
      policy in which each healthy backend is selected in round robin order.
      This is the default. - LEAST_REQUEST: An O(1) algorithm which selects
      two random healthy hosts and picks the host which has fewer active
      requests. - RING_HASH: The ring/modulo hash load balancer implements
      consistent hashing to backends. The algorithm has the property that the
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
      HTTP, HTTPS, or HTTP2, and load_balancing_scheme set to
      INTERNAL_MANAGED. - A global backend service with the
      load_balancing_scheme set to INTERNAL_SELF_MANAGED, INTERNAL_MANAGED, or
      EXTERNAL_MANAGED. If sessionAffinity is not NONE, and this field is not
      set to MAGLEV or RING_HASH, session affinity settings will not take
      effect. Only ROUND_ROBIN and RING_HASH are supported when the backend
      service is referenced by a URL map that is bound to target gRPC proxy
      that has validateForProxyless field set to true.
    logConfig: This field denotes the logging options for the load balancer
      traffic served by this backend service. If logging is enabled, logs will
      be exported to Stackdriver.
    maxStreamDuration: Specifies the default maximum duration (timeout) for
      streams to this service. Duration is computed from the beginning of the
      stream until the response has been completely processed, including all
      retries. A stream that does not complete in this duration is closed. If
      not specified, there will be no timeout limit, i.e. the maximum duration
      is infinite. This value can be overridden in the PathMatcher
      configuration of the UrlMap that references this backend service. This
      field is only allowed when the loadBalancingScheme of the backend
      service is INTERNAL_SELF_MANAGED.
    metadatas: Deployment metadata associated with the resource to be set by a
      GKE hub controller and read by the backend RCTH
    name: Name of the resource. Provided by the client when the resource is
      created. The name must be 1-63 characters long, and comply with RFC1035.
      Specifically, the name must be 1-63 characters long and match the
      regular expression `[a-z]([-a-z0-9]*[a-z0-9])?` which means the first
      character must be a lowercase letter, and all following characters must
      be a dash, lowercase letter, or digit, except the last character, which
      cannot be a dash.
    network: The URL of the network to which this backend service belongs.
      This field can only be specified when the load balancing scheme is set
      to INTERNAL.
    outlierDetection: Settings controlling the ejection of unhealthy backend
      endpoints from the load balancing pool of each individual proxy instance
      that processes the traffic for the given backend service. If not set,
      this feature is considered disabled. Results of the outlier detection
      algorithm (ejection of endpoints from the load balancing pool and
      returning them back to the pool) are executed independently by each
      proxy instance of the load balancer. In most cases, more than one proxy
      instance handles the traffic received by a backend service. Thus, it is
      possible that an unhealthy endpoint is detected and ejected by only some
      of the proxies, and while this happens, other proxies may continue to
      send requests to the same unhealthy endpoint until they detect and eject
      the unhealthy endpoint. Applicable backend endpoints can be: - VM
      instances in an Instance Group - Endpoints in a Zonal NEG (GCE_VM_IP,
      GCE_VM_IP_PORT) - Endpoints in a Hybrid Connectivity NEG
      (NON_GCP_PRIVATE_IP_PORT) - Serverless NEGs, that resolve to Cloud Run,
      App Engine, or Cloud Functions Services - Private Service Connect NEGs,
      that resolve to Google-managed regional API endpoints or managed
      services published using Private Service Connect Applicable backend
      service types can be: - A global backend service with the
      loadBalancingScheme set to INTERNAL_SELF_MANAGED or EXTERNAL_MANAGED. -
      A regional backend service with the serviceProtocol set to HTTP, HTTPS,
      or HTTP2, and loadBalancingScheme set to INTERNAL_MANAGED or
      EXTERNAL_MANAGED. Not supported for Serverless NEGs. Not supported when
      the backend service is referenced by a URL map that is bound to target
      gRPC proxy that has validateForProxyless field set to true.
    port: Deprecated in favor of portName. The TCP port to connect on the
      backend. The default value is 80. For internal passthrough Network Load
      Balancers and external passthrough Network Load Balancers, omit port.
    portName: A named port on a backend instance group representing the port
      for communication to the backend VMs in that group. The named port must
      be [defined on each backend instance
      group](https://cloud.google.com/load-balancing/docs/backend-
      service#named_ports). This parameter has no meaning if the backends are
      NEGs. For internal passthrough Network Load Balancers and external
      passthrough Network Load Balancers, omit port_name.
    protocol: The protocol this BackendService uses to communicate with
      backends. Possible values are HTTP, HTTPS, HTTP2, TCP, SSL, UDP or GRPC.
      depending on the chosen load balancer or Traffic Director configuration.
      Refer to the documentation for the load balancers or for Traffic
      Director for more information. Must be set to GRPC when the backend
      service is referenced by a URL map that is bound to target gRPC proxy.
    region: [Output Only] URL of the region where the regional backend service
      resides. This field is not applicable to global backend services. You
      must specify this field as part of the HTTP request URL. It is not
      settable as a field in the request body.
    securityPolicy: [Output Only] The resource URL for the security policy
      associated with this backend service.
    securitySettings: This field specifies the security settings that apply to
      this backend service. This field is applicable to a global backend
      service with the load_balancing_scheme set to INTERNAL_SELF_MANAGED.
    selfLink: [Output Only] Server-defined URL for the resource.
    serviceBindings: URLs of networkservices.ServiceBinding resources. Can
      only be set if load balancing scheme is INTERNAL_SELF_MANAGED. If set,
      lists of backends and health checks must be both empty.
    serviceLbPolicy: URL to networkservices.ServiceLbPolicy resource. Can only
      be set if load balancing scheme is EXTERNAL, EXTERNAL_MANAGED,
      INTERNAL_MANAGED or INTERNAL_SELF_MANAGED and the scope is global.
    sessionAffinity: Type of session affinity to use. The default is NONE.
      Only NONE and HEADER_FIELD are supported when the backend service is
      referenced by a URL map that is bound to target gRPC proxy that has
      validateForProxyless field set to true. For more details, see: [Session
      Affinity](https://cloud.google.com/load-balancing/docs/backend-
      service#session_affinity).
    subsetting: A Subsetting attribute.
    timeoutSec: The backend service timeout has a different meaning depending
      on the type of load balancer. For more information see, Backend service
      settings. The default is 30 seconds. The full range of timeout values
      allowed goes from 1 through 2,147,483,647 seconds. This value can be
      overridden in the PathMatcher configuration of the UrlMap that
      references this backend service. Not supported when the backend service
      is referenced by a URL map that is bound to target gRPC proxy that has
      validateForProxyless field set to true. Instead, use maxStreamDuration.
    usedBy: A BackendServiceUsedBy attribute.
  """

    class CompressionModeValueValuesEnum(_messages.Enum):
        """Compress text responses using Brotli or gzip compression, based on the
    client's Accept-Encoding header.

    Values:
      AUTOMATIC: Automatically uses the best compression based on the Accept-
        Encoding header sent by the client.
      DISABLED: Disables compression. Existing compressed responses cached by
        Cloud CDN will not be served to clients.
    """
        AUTOMATIC = 0
        DISABLED = 1

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

    class LoadBalancingSchemeValueValuesEnum(_messages.Enum):
        """Specifies the load balancer type. A backend service created for one
    type of load balancer cannot be used with another. For more information,
    refer to Choosing a load balancer.

    Values:
      EXTERNAL: Signifies that this will be used for classic Application Load
        Balancers, global external proxy Network Load Balancers, or external
        passthrough Network Load Balancers.
      EXTERNAL_MANAGED: Signifies that this will be used for global external
        Application Load Balancers, regional external Application Load
        Balancers, or regional external proxy Network Load Balancers.
      INTERNAL: Signifies that this will be used for internal passthrough
        Network Load Balancers.
      INTERNAL_MANAGED: Signifies that this will be used for internal
        Application Load Balancers.
      INTERNAL_SELF_MANAGED: Signifies that this will be used by Traffic
        Director.
      INVALID_LOAD_BALANCING_SCHEME: <no description>
    """
        EXTERNAL = 0
        EXTERNAL_MANAGED = 1
        INTERNAL = 2
        INTERNAL_MANAGED = 3
        INTERNAL_SELF_MANAGED = 4
        INVALID_LOAD_BALANCING_SCHEME = 5

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

    class ProtocolValueValuesEnum(_messages.Enum):
        """The protocol this BackendService uses to communicate with backends.
    Possible values are HTTP, HTTPS, HTTP2, TCP, SSL, UDP or GRPC. depending
    on the chosen load balancer or Traffic Director configuration. Refer to
    the documentation for the load balancers or for Traffic Director for more
    information. Must be set to GRPC when the backend service is referenced by
    a URL map that is bound to target gRPC proxy.

    Values:
      GRPC: gRPC (available for Traffic Director).
      HTTP: <no description>
      HTTP2: HTTP/2 with SSL.
      HTTPS: <no description>
      SSL: TCP proxying with SSL.
      TCP: TCP proxying or TCP pass-through.
      UDP: UDP.
      UNSPECIFIED: If a Backend Service has UNSPECIFIED as its protocol, it
        can be used with any L3/L4 Forwarding Rules.
    """
        GRPC = 0
        HTTP = 1
        HTTP2 = 2
        HTTPS = 3
        SSL = 4
        TCP = 5
        UDP = 6
        UNSPECIFIED = 7

    class SessionAffinityValueValuesEnum(_messages.Enum):
        """Type of session affinity to use. The default is NONE. Only NONE and
    HEADER_FIELD are supported when the backend service is referenced by a URL
    map that is bound to target gRPC proxy that has validateForProxyless field
    set to true. For more details, see: [Session
    Affinity](https://cloud.google.com/load-balancing/docs/backend-
    service#session_affinity).

    Values:
      CLIENT_IP: 2-tuple hash on packet's source and destination IP addresses.
        Connections from the same source IP address to the same destination IP
        address will be served by the same backend VM while that VM remains
        healthy.
      CLIENT_IP_NO_DESTINATION: 1-tuple hash only on packet's source IP
        address. Connections from the same source IP address will be served by
        the same backend VM while that VM remains healthy. This option can
        only be used for Internal TCP/UDP Load Balancing.
      CLIENT_IP_PORT_PROTO: 5-tuple hash on packet's source and destination IP
        addresses, IP protocol, and source and destination ports. Connections
        for the same IP protocol from the same source IP address and port to
        the same destination IP address and port will be served by the same
        backend VM while that VM remains healthy. This option cannot be used
        for HTTP(S) load balancing.
      CLIENT_IP_PROTO: 3-tuple hash on packet's source and destination IP
        addresses, and IP protocol. Connections for the same IP protocol from
        the same source IP address to the same destination IP address will be
        served by the same backend VM while that VM remains healthy. This
        option cannot be used for HTTP(S) load balancing.
      GENERATED_COOKIE: Hash based on a cookie generated by the L7
        loadbalancer. Only valid for HTTP(S) load balancing.
      HEADER_FIELD: The hash is based on a user specified header field.
      HTTP_COOKIE: The hash is based on a user provided cookie.
      NONE: No session affinity. Connections from the same client IP may go to
        any instance in the pool.
    """
        CLIENT_IP = 0
        CLIENT_IP_NO_DESTINATION = 1
        CLIENT_IP_PORT_PROTO = 2
        CLIENT_IP_PROTO = 3
        GENERATED_COOKIE = 4
        HEADER_FIELD = 5
        HTTP_COOKIE = 6
        NONE = 7

    @encoding.MapUnrecognizedFields('additionalProperties')
    class MetadatasValue(_messages.Message):
        """Deployment metadata associated with the resource to be set by a GKE
    hub controller and read by the backend RCTH

    Messages:
      AdditionalProperty: An additional property for a MetadatasValue object.

    Fields:
      additionalProperties: Additional properties of type MetadatasValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a MetadatasValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    affinityCookieTtlSec = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    backends = _messages.MessageField('Backend', 2, repeated=True)
    cdnPolicy = _messages.MessageField('BackendServiceCdnPolicy', 3)
    circuitBreakers = _messages.MessageField('CircuitBreakers', 4)
    compressionMode = _messages.EnumField('CompressionModeValueValuesEnum', 5)
    connectionDraining = _messages.MessageField('ConnectionDraining', 6)
    connectionTrackingPolicy = _messages.MessageField('BackendServiceConnectionTrackingPolicy', 7)
    consistentHash = _messages.MessageField('ConsistentHashLoadBalancerSettings', 8)
    creationTimestamp = _messages.StringField(9)
    customRequestHeaders = _messages.StringField(10, repeated=True)
    customResponseHeaders = _messages.StringField(11, repeated=True)
    description = _messages.StringField(12)
    edgeSecurityPolicy = _messages.StringField(13)
    enableCDN = _messages.BooleanField(14)
    failoverPolicy = _messages.MessageField('BackendServiceFailoverPolicy', 15)
    fingerprint = _messages.BytesField(16)
    healthChecks = _messages.StringField(17, repeated=True)
    iap = _messages.MessageField('BackendServiceIAP', 18)
    id = _messages.IntegerField(19, variant=_messages.Variant.UINT64)
    ipAddressSelectionPolicy = _messages.EnumField('IpAddressSelectionPolicyValueValuesEnum', 20)
    kind = _messages.StringField(21, default='compute#backendService')
    loadBalancingScheme = _messages.EnumField('LoadBalancingSchemeValueValuesEnum', 22)
    localityLbPolicies = _messages.MessageField('BackendServiceLocalityLoadBalancingPolicyConfig', 23, repeated=True)
    localityLbPolicy = _messages.EnumField('LocalityLbPolicyValueValuesEnum', 24)
    logConfig = _messages.MessageField('BackendServiceLogConfig', 25)
    maxStreamDuration = _messages.MessageField('Duration', 26)
    metadatas = _messages.MessageField('MetadatasValue', 27)
    name = _messages.StringField(28)
    network = _messages.StringField(29)
    outlierDetection = _messages.MessageField('OutlierDetection', 30)
    port = _messages.IntegerField(31, variant=_messages.Variant.INT32)
    portName = _messages.StringField(32)
    protocol = _messages.EnumField('ProtocolValueValuesEnum', 33)
    region = _messages.StringField(34)
    securityPolicy = _messages.StringField(35)
    securitySettings = _messages.MessageField('SecuritySettings', 36)
    selfLink = _messages.StringField(37)
    serviceBindings = _messages.StringField(38, repeated=True)
    serviceLbPolicy = _messages.StringField(39)
    sessionAffinity = _messages.EnumField('SessionAffinityValueValuesEnum', 40)
    subsetting = _messages.MessageField('Subsetting', 41)
    timeoutSec = _messages.IntegerField(42, variant=_messages.Variant.INT32)
    usedBy = _messages.MessageField('BackendServiceUsedBy', 43, repeated=True)