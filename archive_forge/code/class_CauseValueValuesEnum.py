from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CauseValueValuesEnum(_messages.Enum):
    """Cause that the packet is dropped.

    Values:
      CAUSE_UNSPECIFIED: Cause is unspecified.
      UNKNOWN_EXTERNAL_ADDRESS: Destination external address cannot be
        resolved to a known target. If the address is used in a Google Cloud
        project, provide the project ID as test input.
      FOREIGN_IP_DISALLOWED: A Compute Engine instance can only send or
        receive a packet with a foreign IP address if ip_forward is enabled.
      FIREWALL_RULE: Dropped due to a firewall rule, unless allowed due to
        connection tracking.
      NO_ROUTE: Dropped due to no matching routes.
      ROUTE_BLACKHOLE: Dropped due to invalid route. Route's next hop is a
        blackhole.
      ROUTE_WRONG_NETWORK: Packet is sent to a wrong (unintended) network.
        Example: you trace a packet from VM1:Network1 to VM2:Network2,
        however, the route configured in Network1 sends the packet destined
        for VM2's IP address to Network3.
      ROUTE_NEXT_HOP_IP_ADDRESS_NOT_RESOLVED: Route's next hop IP address
        cannot be resolved to a GCP resource.
      ROUTE_NEXT_HOP_RESOURCE_NOT_FOUND: Route's next hop resource is not
        found.
      ROUTE_NEXT_HOP_INSTANCE_WRONG_NETWORK: Route's next hop instance doesn't
        have a NIC in the route's network.
      ROUTE_NEXT_HOP_INSTANCE_NON_PRIMARY_IP: Route's next hop IP address is
        not a primary IP address of the next hop instance.
      ROUTE_NEXT_HOP_FORWARDING_RULE_IP_MISMATCH: Route's next hop forwarding
        rule doesn't match next hop IP address.
      ROUTE_NEXT_HOP_VPN_TUNNEL_NOT_ESTABLISHED: Route's next hop VPN tunnel
        is down (does not have valid IKE SAs).
      ROUTE_NEXT_HOP_FORWARDING_RULE_TYPE_INVALID: Route's next hop forwarding
        rule type is invalid (it's not a forwarding rule of the internal
        passthrough load balancer).
      NO_ROUTE_FROM_INTERNET_TO_PRIVATE_IPV6_ADDRESS: Packet is sent from the
        Internet to the private IPv6 address.
      VPN_TUNNEL_LOCAL_SELECTOR_MISMATCH: The packet does not match a policy-
        based VPN tunnel local selector.
      VPN_TUNNEL_REMOTE_SELECTOR_MISMATCH: The packet does not match a policy-
        based VPN tunnel remote selector.
      PRIVATE_TRAFFIC_TO_INTERNET: Packet with internal destination address
        sent to the internet gateway.
      PRIVATE_GOOGLE_ACCESS_DISALLOWED: Instance with only an internal IP
        address tries to access Google API and services, but private Google
        access is not enabled in the subnet.
      PRIVATE_GOOGLE_ACCESS_VIA_VPN_TUNNEL_UNSUPPORTED: Source endpoint tries
        to access Google API and services through the VPN tunnel to another
        network, but Private Google Access needs to be enabled in the source
        endpoint network.
      NO_EXTERNAL_ADDRESS: Instance with only an internal IP address tries to
        access external hosts, but Cloud NAT is not enabled in the subnet,
        unless special configurations on a VM allow this connection.
      UNKNOWN_INTERNAL_ADDRESS: Destination internal address cannot be
        resolved to a known target. If this is a shared VPC scenario, verify
        if the service project ID is provided as test input. Otherwise, verify
        if the IP address is being used in the project.
      FORWARDING_RULE_MISMATCH: Forwarding rule's protocol and ports do not
        match the packet header.
      FORWARDING_RULE_NO_INSTANCES: Forwarding rule does not have backends
        configured.
      FIREWALL_BLOCKING_LOAD_BALANCER_BACKEND_HEALTH_CHECK: Firewalls block
        the health check probes to the backends and cause the backends to be
        unavailable for traffic from the load balancer. For more details, see
        [Health check firewall rules](https://cloud.google.com/load-
        balancing/docs/health-checks#firewall_rules).
      INSTANCE_NOT_RUNNING: Packet is sent from or to a Compute Engine
        instance that is not in a running state.
      GKE_CLUSTER_NOT_RUNNING: Packet sent from or to a GKE cluster that is
        not in running state.
      CLOUD_SQL_INSTANCE_NOT_RUNNING: Packet sent from or to a Cloud SQL
        instance that is not in running state.
      TRAFFIC_TYPE_BLOCKED: The type of traffic is blocked and the user cannot
        configure a firewall rule to enable it. See [Always blocked
        traffic](https://cloud.google.com/vpc/docs/firewalls#blockedtraffic)
        for more details.
      GKE_MASTER_UNAUTHORIZED_ACCESS: Access to Google Kubernetes Engine
        cluster master's endpoint is not authorized. See [Access to the
        cluster endpoints](https://cloud.google.com/kubernetes-
        engine/docs/how-to/private-clusters#access_to_the_cluster_endpoints)
        for more details.
      CLOUD_SQL_INSTANCE_UNAUTHORIZED_ACCESS: Access to the Cloud SQL instance
        endpoint is not authorized. See [Authorizing with authorized
        networks](https://cloud.google.com/sql/docs/mysql/authorize-networks)
        for more details.
      DROPPED_INSIDE_GKE_SERVICE: Packet was dropped inside Google Kubernetes
        Engine Service.
      DROPPED_INSIDE_CLOUD_SQL_SERVICE: Packet was dropped inside Cloud SQL
        Service.
      GOOGLE_MANAGED_SERVICE_NO_PEERING: Packet was dropped because there is
        no peering between the originating network and the Google Managed
        Services Network.
      GOOGLE_MANAGED_SERVICE_NO_PSC_ENDPOINT: Packet was dropped because the
        Google-managed service uses Private Service Connect (PSC), but the PSC
        endpoint is not found in the project.
      GKE_PSC_ENDPOINT_MISSING: Packet was dropped because the GKE cluster
        uses Private Service Connect (PSC), but the PSC endpoint is not found
        in the project.
      CLOUD_SQL_INSTANCE_NO_IP_ADDRESS: Packet was dropped because the Cloud
        SQL instance has neither a private nor a public IP address.
      GKE_CONTROL_PLANE_REGION_MISMATCH: Packet was dropped because a GKE
        cluster private endpoint is unreachable from a region different from
        the cluster's region.
      PUBLIC_GKE_CONTROL_PLANE_TO_PRIVATE_DESTINATION: Packet sent from a
        public GKE cluster control plane to a private IP address.
      GKE_CONTROL_PLANE_NO_ROUTE: Packet was dropped because there is no route
        from a GKE cluster control plane to a destination network.
      CLOUD_SQL_INSTANCE_NOT_CONFIGURED_FOR_EXTERNAL_TRAFFIC: Packet sent from
        a Cloud SQL instance to an external IP address is not allowed. The
        Cloud SQL instance is not configured to send packets to external IP
        addresses.
      PUBLIC_CLOUD_SQL_INSTANCE_TO_PRIVATE_DESTINATION: Packet sent from a
        Cloud SQL instance with only a public IP address to a private IP
        address.
      CLOUD_SQL_INSTANCE_NO_ROUTE: Packet was dropped because there is no
        route from a Cloud SQL instance to a destination network.
      CLOUD_FUNCTION_NOT_ACTIVE: Packet could be dropped because the Cloud
        Function is not in an active status.
      VPC_CONNECTOR_NOT_SET: Packet could be dropped because no VPC connector
        is set.
      VPC_CONNECTOR_NOT_RUNNING: Packet could be dropped because the VPC
        connector is not in a running state.
      FORWARDING_RULE_REGION_MISMATCH: Packet could be dropped because it was
        sent from a different region to a regional forwarding without global
        access.
      PSC_CONNECTION_NOT_ACCEPTED: The Private Service Connect endpoint is in
        a project that is not approved to connect to the service.
      PSC_ENDPOINT_ACCESSED_FROM_PEERED_NETWORK: The packet is sent to the
        Private Service Connect endpoint over the peering, but [it's not
        supported](https://cloud.google.com/vpc/docs/configure-private-
        service-connect-services#on-premises).
      PSC_NEG_PRODUCER_ENDPOINT_NO_GLOBAL_ACCESS: The packet is sent to the
        Private Service Connect backend (network endpoint group), but the
        producer PSC forwarding rule does not have global access enabled.
      PSC_NEG_PRODUCER_FORWARDING_RULE_MULTIPLE_PORTS: The packet is sent to
        the Private Service Connect backend (network endpoint group), but the
        producer PSC forwarding rule has multiple ports specified.
      CLOUD_SQL_PSC_NEG_UNSUPPORTED: The packet is sent to the Private Service
        Connect backend (network endpoint group) targeting a Cloud SQL service
        attachment, but this configuration is not supported.
      NO_NAT_SUBNETS_FOR_PSC_SERVICE_ATTACHMENT: No NAT subnets are defined
        for the PSC service attachment.
      HYBRID_NEG_NON_DYNAMIC_ROUTE_MATCHED: The packet sent from the hybrid
        NEG proxy matches a non-dynamic route, but such a configuration is not
        supported.
      HYBRID_NEG_NON_LOCAL_DYNAMIC_ROUTE_MATCHED: The packet sent from the
        hybrid NEG proxy matches a dynamic route with a next hop in a
        different region, but such a configuration is not supported.
      CLOUD_RUN_REVISION_NOT_READY: Packet sent from a Cloud Run revision that
        is not ready.
      DROPPED_INSIDE_PSC_SERVICE_PRODUCER: Packet was dropped inside Private
        Service Connect service producer.
      LOAD_BALANCER_HAS_NO_PROXY_SUBNET: Packet sent to a load balancer, which
        requires a proxy-only subnet and the subnet is not found.
      CLOUD_NAT_NO_ADDRESSES: Packet sent to Cloud Nat without active NAT IPs.
      ROUTING_LOOP: Packet is stuck in a routing loop.
    """
    CAUSE_UNSPECIFIED = 0
    UNKNOWN_EXTERNAL_ADDRESS = 1
    FOREIGN_IP_DISALLOWED = 2
    FIREWALL_RULE = 3
    NO_ROUTE = 4
    ROUTE_BLACKHOLE = 5
    ROUTE_WRONG_NETWORK = 6
    ROUTE_NEXT_HOP_IP_ADDRESS_NOT_RESOLVED = 7
    ROUTE_NEXT_HOP_RESOURCE_NOT_FOUND = 8
    ROUTE_NEXT_HOP_INSTANCE_WRONG_NETWORK = 9
    ROUTE_NEXT_HOP_INSTANCE_NON_PRIMARY_IP = 10
    ROUTE_NEXT_HOP_FORWARDING_RULE_IP_MISMATCH = 11
    ROUTE_NEXT_HOP_VPN_TUNNEL_NOT_ESTABLISHED = 12
    ROUTE_NEXT_HOP_FORWARDING_RULE_TYPE_INVALID = 13
    NO_ROUTE_FROM_INTERNET_TO_PRIVATE_IPV6_ADDRESS = 14
    VPN_TUNNEL_LOCAL_SELECTOR_MISMATCH = 15
    VPN_TUNNEL_REMOTE_SELECTOR_MISMATCH = 16
    PRIVATE_TRAFFIC_TO_INTERNET = 17
    PRIVATE_GOOGLE_ACCESS_DISALLOWED = 18
    PRIVATE_GOOGLE_ACCESS_VIA_VPN_TUNNEL_UNSUPPORTED = 19
    NO_EXTERNAL_ADDRESS = 20
    UNKNOWN_INTERNAL_ADDRESS = 21
    FORWARDING_RULE_MISMATCH = 22
    FORWARDING_RULE_NO_INSTANCES = 23
    FIREWALL_BLOCKING_LOAD_BALANCER_BACKEND_HEALTH_CHECK = 24
    INSTANCE_NOT_RUNNING = 25
    GKE_CLUSTER_NOT_RUNNING = 26
    CLOUD_SQL_INSTANCE_NOT_RUNNING = 27
    TRAFFIC_TYPE_BLOCKED = 28
    GKE_MASTER_UNAUTHORIZED_ACCESS = 29
    CLOUD_SQL_INSTANCE_UNAUTHORIZED_ACCESS = 30
    DROPPED_INSIDE_GKE_SERVICE = 31
    DROPPED_INSIDE_CLOUD_SQL_SERVICE = 32
    GOOGLE_MANAGED_SERVICE_NO_PEERING = 33
    GOOGLE_MANAGED_SERVICE_NO_PSC_ENDPOINT = 34
    GKE_PSC_ENDPOINT_MISSING = 35
    CLOUD_SQL_INSTANCE_NO_IP_ADDRESS = 36
    GKE_CONTROL_PLANE_REGION_MISMATCH = 37
    PUBLIC_GKE_CONTROL_PLANE_TO_PRIVATE_DESTINATION = 38
    GKE_CONTROL_PLANE_NO_ROUTE = 39
    CLOUD_SQL_INSTANCE_NOT_CONFIGURED_FOR_EXTERNAL_TRAFFIC = 40
    PUBLIC_CLOUD_SQL_INSTANCE_TO_PRIVATE_DESTINATION = 41
    CLOUD_SQL_INSTANCE_NO_ROUTE = 42
    CLOUD_FUNCTION_NOT_ACTIVE = 43
    VPC_CONNECTOR_NOT_SET = 44
    VPC_CONNECTOR_NOT_RUNNING = 45
    FORWARDING_RULE_REGION_MISMATCH = 46
    PSC_CONNECTION_NOT_ACCEPTED = 47
    PSC_ENDPOINT_ACCESSED_FROM_PEERED_NETWORK = 48
    PSC_NEG_PRODUCER_ENDPOINT_NO_GLOBAL_ACCESS = 49
    PSC_NEG_PRODUCER_FORWARDING_RULE_MULTIPLE_PORTS = 50
    CLOUD_SQL_PSC_NEG_UNSUPPORTED = 51
    NO_NAT_SUBNETS_FOR_PSC_SERVICE_ATTACHMENT = 52
    HYBRID_NEG_NON_DYNAMIC_ROUTE_MATCHED = 53
    HYBRID_NEG_NON_LOCAL_DYNAMIC_ROUTE_MATCHED = 54
    CLOUD_RUN_REVISION_NOT_READY = 55
    DROPPED_INSIDE_PSC_SERVICE_PRODUCER = 56
    LOAD_BALANCER_HAS_NO_PROXY_SUBNET = 57
    CLOUD_NAT_NO_ADDRESSES = 58
    ROUTING_LOOP = 59