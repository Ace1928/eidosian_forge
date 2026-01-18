from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ClusterUpdate(_messages.Message):
    """ClusterUpdate describes an update to the cluster. Exactly one update can
  be applied to a cluster with each request, so at most one field can be
  provided.

  Enums:
    DesiredDatapathProviderValueValuesEnum: The desired datapath provider for
      the cluster.
    DesiredInTransitEncryptionConfigValueValuesEnum: Specify the details of
      in-transit encryption.
    DesiredPrivateIpv6GoogleAccessValueValuesEnum: The desired state of IPv6
      connectivity to Google Services.
    DesiredStackTypeValueValuesEnum: The desired stack type of the cluster. If
      a stack type is provided and does not match the current stack type of
      the cluster, update will attempt to change the stack type to the new
      type.

  Fields:
    additionalPodRangesConfig: The additional pod ranges to be added to the
      cluster. These pod ranges can be used by node pools to allocate pod IPs.
    desiredAddonsConfig: Configurations for the various addons available to
      run in the cluster.
    desiredAuthenticatorGroupsConfig: The desired authenticator groups config
      for the cluster.
    desiredAutopilot: The desired Autopilot configuration for the cluster.
    desiredAutopilotInsecureKubeletReadonlyPortEnabled: Enable/disable kubelet
      readonly port for autopilot cluster
    desiredAutopilotWorkloadPolicyConfig: The desired workload policy
      configuration for the autopilot cluster.
    desiredBinaryAuthorization: The desired configuration options for the
      Binary Authorization feature.
    desiredClusterAutoscaling: Cluster-level autoscaling configuration.
    desiredCompliancePostureConfig: Enable/Disable Compliance Posture features
      for the cluster.
    desiredConcurrentOpsConfig: Desired value for the cluster's
      concurrent_ops_config.
    desiredContainerdConfig: The desired containerd config for the cluster.
    desiredControlPlaneEndpointsConfig: Control plane endpoints configuration.
    desiredCostManagementConfig: The desired configuration for the fine-
      grained cost management feature.
    desiredDatabaseEncryption: Configuration of etcd encryption.
    desiredDatapathProvider: The desired datapath provider for the cluster.
    desiredDefaultEnablePrivateNodes: Override the default setting of whether
      furture created nodes have private IP addresses only, namely
      NetworkConfig.default_enable_private_nodes
    desiredDefaultSnatStatus: The desired status of whether to disable default
      sNAT for this cluster.
    desiredDnsConfig: DNSConfig contains clusterDNS config for this cluster.
    desiredEnableCiliumClusterwideNetworkPolicy: Enable/Disable Cilium
      Clusterwide Network Policy for the cluster.
    desiredEnableFqdnNetworkPolicy: Enable/Disable FQDN Network Policy for the
      cluster.
    desiredEnableMultiNetworking: Enable/Disable Multi-Networking for the
      cluster
    desiredEnablePrivateEndpoint: Enable/Disable private endpoint for the
      cluster's master.
    desiredFleet: The desired fleet configuration for the cluster.
    desiredGatewayApiConfig: The desired config of Gateway API on this
      cluster.
    desiredGcfsConfig: The desired GCFS config for the cluster
    desiredIdentityServiceConfig: The desired Identity Service component
      configuration.
    desiredImage: The desired name of the image to use for this node. This is
      used to create clusters using a custom image. NOTE: Set the
      "desired_node_pool" field as well.
    desiredImageProject: The project containing the desired image to use for
      this node. This is used to create clusters using a custom image. NOTE:
      Set the "desired_node_pool" field as well.
    desiredImageType: The desired image type for the node pool. NOTE: Set the
      "desired_node_pool" field as well.
    desiredInTransitEncryptionConfig: Specify the details of in-transit
      encryption.
    desiredIntraNodeVisibilityConfig: The desired config of Intra-node
      visibility.
    desiredK8sBetaApis: Desired Beta APIs to be enabled for cluster.
    desiredL4ilbSubsettingConfig: The desired L4 Internal Load Balancer
      Subsetting configuration.
    desiredLocations: The desired list of Google Compute Engine
      [zones](https://cloud.google.com/compute/docs/zones#available) in which
      the cluster's nodes should be located. This list must always include the
      cluster's primary zone. Warning: changing cluster locations will update
      the locations of all node pools and will result in nodes being added
      and/or removed.
    desiredLoggingConfig: The desired logging configuration.
    desiredLoggingService: The logging service the cluster should use to write
      logs. Currently available options: * `logging.googleapis.com/kubernetes`
      - The Cloud Logging service with a Kubernetes-native resource model *
      `logging.googleapis.com` - The legacy Cloud Logging service (no longer
      available as of GKE 1.15). * `none` - no logs will be exported from the
      cluster. If left as an empty string,`logging.googleapis.com/kubernetes`
      will be used for GKE 1.14+ or `logging.googleapis.com` for earlier
      versions.
    desiredManagedConfig: The desired managed config for the cluster.
    desiredMasterAuthorizedNetworksConfig: The desired configuration options
      for master authorized networks feature.
    desiredMasterVersion: The Kubernetes version to change the master to.
      Users may specify either explicit versions offered by Kubernetes Engine
      or version aliases, which have the following behavior: - "latest": picks
      the highest valid Kubernetes version - "1.X": picks the highest valid
      patch+gke.N patch in the 1.X version - "1.X.Y": picks the highest valid
      gke.N patch in the 1.X.Y version - "1.X.Y-gke.N": picks an explicit
      Kubernetes version - "-": picks the default Kubernetes version
    desiredMeshCertificates: Configuration for issuance of mTLS keys and
      certificates to Kubernetes pods.
    desiredMonitoringConfig: The desired monitoring configuration.
    desiredMonitoringService: The monitoring service the cluster should use to
      write metrics. Currently available options: *
      "monitoring.googleapis.com/kubernetes" - The Cloud Monitoring service
      with a Kubernetes-native resource model * `monitoring.googleapis.com` -
      The legacy Cloud Monitoring service (no longer available as of GKE
      1.15). * `none` - No metrics will be exported from the cluster. If left
      as an empty string,`monitoring.googleapis.com/kubernetes` will be used
      for GKE 1.14+ or `monitoring.googleapis.com` for earlier versions.
    desiredNetworkPerformanceConfig: The desired network performance config.
    desiredNodeKubeletConfig: The desired node kubelet config for the cluster.
    desiredNodePoolAutoConfigKubeletConfig: The desired node kubelet config
      for all auto-provisioned node pools in autopilot clusters and node auto-
      provisioning enabled clusters.
    desiredNodePoolAutoConfigNetworkTags: The desired network tags that apply
      to all auto-provisioned node pools in autopilot clusters and node auto-
      provisioning enabled clusters.
    desiredNodePoolAutoConfigResourceManagerTags: The desired resource manager
      tags that apply to all auto-provisioned node pools in autopilot clusters
      and node auto-provisioning enabled clusters.
    desiredNodePoolAutoscaling: Autoscaler configuration for the node pool
      specified in desired_node_pool_id. If there is only one pool in the
      cluster and desired_node_pool_id is not provided then the change applies
      to that single node pool.
    desiredNodePoolId: The node pool to be upgraded. This field is mandatory
      if "desired_node_version", "desired_image_family" or
      "desired_node_pool_autoscaling" is specified and there is more than one
      node pool on the cluster.
    desiredNodePoolLoggingConfig: The desired node pool logging configuration
      defaults for the cluster.
    desiredNodeVersion: The Kubernetes version to change the nodes to
      (typically an upgrade). Users may specify either explicit versions
      offered by Kubernetes Engine or version aliases, which have the
      following behavior: - "latest": picks the highest valid Kubernetes
      version - "1.X": picks the highest valid patch+gke.N patch in the 1.X
      version - "1.X.Y": picks the highest valid gke.N patch in the 1.X.Y
      version - "1.X.Y-gke.N": picks an explicit Kubernetes version - "-":
      picks the Kubernetes master version
    desiredNotificationConfig: The desired notification configuration.
    desiredParentProductConfig: The desired parent product config for the
      cluster.
    desiredPrivateClusterConfig: The desired private cluster configuration.
      master_global_access_config is the only field that can be changed via
      this field. See also ClusterUpdate.desired_enable_private_endpoint for
      modifying other fields within PrivateClusterConfig.
    desiredPrivateIpv6GoogleAccess: The desired state of IPv6 connectivity to
      Google Services.
    desiredReleaseChannel: The desired release channel configuration.
    desiredResourceUsageExportConfig: The desired configuration for exporting
      resource usage.
    desiredRuntimeVulnerabilityInsightConfig: Enable/Disable RVI features for
      the cluster.
    desiredSecretManagerConfig: Enable/Disable Secret Manager Config.
    desiredSecurityPostureConfig: Enable/Disable Security Posture API features
      for the cluster.
    desiredServiceExternalIpsConfig: ServiceExternalIPsConfig specifies the
      config for the use of Services with ExternalIPs field.
    desiredShieldedNodes: Configuration for Shielded Nodes.
    desiredStackType: The desired stack type of the cluster. If a stack type
      is provided and does not match the current stack type of the cluster,
      update will attempt to change the stack type to the new type.
    desiredVerticalPodAutoscaling: Cluster-level Vertical Pod Autoscaling
      configuration.
    desiredWorkloadIdentityConfig: Configuration for Workload Identity.
    enableK8sBetaApis: Kubernetes open source beta apis enabled on the
      cluster. Only beta apis
    etag: The current etag of the cluster. If an etag is provided and does not
      match the current etag of the cluster, update will be blocked and an
      ABORTED error will be returned.
    removedAdditionalPodRangesConfig: The additional pod ranges that are to be
      removed from the cluster. The pod ranges specified here must have been
      specified earlier in the 'additional_pod_ranges_config' argument.
  """

    class DesiredDatapathProviderValueValuesEnum(_messages.Enum):
        """The desired datapath provider for the cluster.

    Values:
      DATAPATH_PROVIDER_UNSPECIFIED: Default value.
      LEGACY_DATAPATH: Use the IPTables implementation based on kube-proxy.
      ADVANCED_DATAPATH: Use the eBPF based GKE Dataplane V2 with additional
        features. See the [GKE Dataplane V2
        documentation](https://cloud.google.com/kubernetes-engine/docs/how-
        to/dataplane-v2) for more.
      MIGRATE_TO_ADVANCED_DATAPATH: Cluster has some existing nodes but new
        nodes should use ADVANCED_DATAPATH.
      MIGRATE_TO_LEGACY_DATAPATH: Cluster has some existing nodes but new
        nodes should use LEGACY_DATAPATH.
    """
        DATAPATH_PROVIDER_UNSPECIFIED = 0
        LEGACY_DATAPATH = 1
        ADVANCED_DATAPATH = 2
        MIGRATE_TO_ADVANCED_DATAPATH = 3
        MIGRATE_TO_LEGACY_DATAPATH = 4

    class DesiredInTransitEncryptionConfigValueValuesEnum(_messages.Enum):
        """Specify the details of in-transit encryption.

    Values:
      IN_TRANSIT_ENCRYPTION_CONFIG_UNSPECIFIED: Unspecified, will be inferred
        as default - IN_TRANSIT_ENCRYPTION_UNSPECIFIED.
      IN_TRANSIT_ENCRYPTION_DISABLED: In-transit encryption is disabled.
      IN_TRANSIT_ENCRYPTION_INTER_NODE_TRANSPARENT: Data in-transit is
        encrypted using inter-node transparent encryption.
    """
        IN_TRANSIT_ENCRYPTION_CONFIG_UNSPECIFIED = 0
        IN_TRANSIT_ENCRYPTION_DISABLED = 1
        IN_TRANSIT_ENCRYPTION_INTER_NODE_TRANSPARENT = 2

    class DesiredPrivateIpv6GoogleAccessValueValuesEnum(_messages.Enum):
        """The desired state of IPv6 connectivity to Google Services.

    Values:
      PRIVATE_IPV6_GOOGLE_ACCESS_UNSPECIFIED: Default value. Same as DISABLED
      PRIVATE_IPV6_GOOGLE_ACCESS_DISABLED: No private access to or from Google
        Services
      PRIVATE_IPV6_GOOGLE_ACCESS_TO_GOOGLE: Enables private IPv6 access to
        Google Services from GKE
      PRIVATE_IPV6_GOOGLE_ACCESS_BIDIRECTIONAL: Enables private IPv6 access to
        and from Google Services
    """
        PRIVATE_IPV6_GOOGLE_ACCESS_UNSPECIFIED = 0
        PRIVATE_IPV6_GOOGLE_ACCESS_DISABLED = 1
        PRIVATE_IPV6_GOOGLE_ACCESS_TO_GOOGLE = 2
        PRIVATE_IPV6_GOOGLE_ACCESS_BIDIRECTIONAL = 3

    class DesiredStackTypeValueValuesEnum(_messages.Enum):
        """The desired stack type of the cluster. If a stack type is provided and
    does not match the current stack type of the cluster, update will attempt
    to change the stack type to the new type.

    Values:
      STACK_TYPE_UNSPECIFIED: Default value, will be defaulted as IPV4 only
      IPV4: Cluster is IPV4 only
      IPV4_IPV6: Cluster can use both IPv4 and IPv6
    """
        STACK_TYPE_UNSPECIFIED = 0
        IPV4 = 1
        IPV4_IPV6 = 2
    additionalPodRangesConfig = _messages.MessageField('AdditionalPodRangesConfig', 1)
    desiredAddonsConfig = _messages.MessageField('AddonsConfig', 2)
    desiredAuthenticatorGroupsConfig = _messages.MessageField('AuthenticatorGroupsConfig', 3)
    desiredAutopilot = _messages.MessageField('Autopilot', 4)
    desiredAutopilotInsecureKubeletReadonlyPortEnabled = _messages.BooleanField(5)
    desiredAutopilotWorkloadPolicyConfig = _messages.MessageField('WorkloadPolicyConfig', 6)
    desiredBinaryAuthorization = _messages.MessageField('BinaryAuthorization', 7)
    desiredClusterAutoscaling = _messages.MessageField('ClusterAutoscaling', 8)
    desiredCompliancePostureConfig = _messages.MessageField('CompliancePostureConfig', 9)
    desiredConcurrentOpsConfig = _messages.MessageField('ConcurrentOpsConfig', 10)
    desiredContainerdConfig = _messages.MessageField('ContainerdConfig', 11)
    desiredControlPlaneEndpointsConfig = _messages.MessageField('ControlPlaneEndpointsConfig', 12)
    desiredCostManagementConfig = _messages.MessageField('CostManagementConfig', 13)
    desiredDatabaseEncryption = _messages.MessageField('DatabaseEncryption', 14)
    desiredDatapathProvider = _messages.EnumField('DesiredDatapathProviderValueValuesEnum', 15)
    desiredDefaultEnablePrivateNodes = _messages.BooleanField(16)
    desiredDefaultSnatStatus = _messages.MessageField('DefaultSnatStatus', 17)
    desiredDnsConfig = _messages.MessageField('DNSConfig', 18)
    desiredEnableCiliumClusterwideNetworkPolicy = _messages.BooleanField(19)
    desiredEnableFqdnNetworkPolicy = _messages.BooleanField(20)
    desiredEnableMultiNetworking = _messages.BooleanField(21)
    desiredEnablePrivateEndpoint = _messages.BooleanField(22)
    desiredFleet = _messages.MessageField('Fleet', 23)
    desiredGatewayApiConfig = _messages.MessageField('GatewayAPIConfig', 24)
    desiredGcfsConfig = _messages.MessageField('GcfsConfig', 25)
    desiredIdentityServiceConfig = _messages.MessageField('IdentityServiceConfig', 26)
    desiredImage = _messages.StringField(27)
    desiredImageProject = _messages.StringField(28)
    desiredImageType = _messages.StringField(29)
    desiredInTransitEncryptionConfig = _messages.EnumField('DesiredInTransitEncryptionConfigValueValuesEnum', 30)
    desiredIntraNodeVisibilityConfig = _messages.MessageField('IntraNodeVisibilityConfig', 31)
    desiredK8sBetaApis = _messages.MessageField('K8sBetaAPIConfig', 32)
    desiredL4ilbSubsettingConfig = _messages.MessageField('ILBSubsettingConfig', 33)
    desiredLocations = _messages.StringField(34, repeated=True)
    desiredLoggingConfig = _messages.MessageField('LoggingConfig', 35)
    desiredLoggingService = _messages.StringField(36)
    desiredManagedConfig = _messages.MessageField('ManagedConfig', 37)
    desiredMasterAuthorizedNetworksConfig = _messages.MessageField('MasterAuthorizedNetworksConfig', 38)
    desiredMasterVersion = _messages.StringField(39)
    desiredMeshCertificates = _messages.MessageField('MeshCertificates', 40)
    desiredMonitoringConfig = _messages.MessageField('MonitoringConfig', 41)
    desiredMonitoringService = _messages.StringField(42)
    desiredNetworkPerformanceConfig = _messages.MessageField('ClusterNetworkPerformanceConfig', 43)
    desiredNodeKubeletConfig = _messages.MessageField('NodeKubeletConfig', 44)
    desiredNodePoolAutoConfigKubeletConfig = _messages.MessageField('NodeKubeletConfig', 45)
    desiredNodePoolAutoConfigNetworkTags = _messages.MessageField('NetworkTags', 46)
    desiredNodePoolAutoConfigResourceManagerTags = _messages.MessageField('ResourceManagerTags', 47)
    desiredNodePoolAutoscaling = _messages.MessageField('NodePoolAutoscaling', 48)
    desiredNodePoolId = _messages.StringField(49)
    desiredNodePoolLoggingConfig = _messages.MessageField('NodePoolLoggingConfig', 50)
    desiredNodeVersion = _messages.StringField(51)
    desiredNotificationConfig = _messages.MessageField('NotificationConfig', 52)
    desiredParentProductConfig = _messages.MessageField('ParentProductConfig', 53)
    desiredPrivateClusterConfig = _messages.MessageField('PrivateClusterConfig', 54)
    desiredPrivateIpv6GoogleAccess = _messages.EnumField('DesiredPrivateIpv6GoogleAccessValueValuesEnum', 55)
    desiredReleaseChannel = _messages.MessageField('ReleaseChannel', 56)
    desiredResourceUsageExportConfig = _messages.MessageField('ResourceUsageExportConfig', 57)
    desiredRuntimeVulnerabilityInsightConfig = _messages.MessageField('RuntimeVulnerabilityInsightConfig', 58)
    desiredSecretManagerConfig = _messages.MessageField('SecretManagerConfig', 59)
    desiredSecurityPostureConfig = _messages.MessageField('SecurityPostureConfig', 60)
    desiredServiceExternalIpsConfig = _messages.MessageField('ServiceExternalIPsConfig', 61)
    desiredShieldedNodes = _messages.MessageField('ShieldedNodes', 62)
    desiredStackType = _messages.EnumField('DesiredStackTypeValueValuesEnum', 63)
    desiredVerticalPodAutoscaling = _messages.MessageField('VerticalPodAutoscaling', 64)
    desiredWorkloadIdentityConfig = _messages.MessageField('WorkloadIdentityConfig', 65)
    enableK8sBetaApis = _messages.MessageField('K8sBetaAPIConfig', 66)
    etag = _messages.StringField(67)
    removedAdditionalPodRangesConfig = _messages.MessageField('AdditionalPodRangesConfig', 68)