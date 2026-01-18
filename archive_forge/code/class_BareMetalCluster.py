from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BareMetalCluster(_messages.Message):
    """Resource that represents a bare metal user cluster.

  Enums:
    StateValueValuesEnum: Output only. The current state of the bare metal
      user cluster.

  Messages:
    AnnotationsValue: Annotations on the bare metal user cluster. This field
      has the same restrictions as Kubernetes annotations. The total size of
      all keys and values combined is limited to 256k. Key can have 2
      segments: prefix (optional) and name (required), separated by a slash
      (/). Prefix must be a DNS subdomain. Name must be 63 characters or less,
      begin and end with alphanumerics, with dashes (-), underscores (_), dots
      (.), and alphanumerics between.

  Fields:
    adminClusterMembership: Required. The admin cluster this bare metal user
      cluster belongs to. This is the full resource name of the admin
      cluster's fleet membership.
    adminClusterName: Output only. The resource name of the bare metal admin
      cluster managing this user cluster.
    annotations: Annotations on the bare metal user cluster. This field has
      the same restrictions as Kubernetes annotations. The total size of all
      keys and values combined is limited to 256k. Key can have 2 segments:
      prefix (optional) and name (required), separated by a slash (/). Prefix
      must be a DNS subdomain. Name must be 63 characters or less, begin and
      end with alphanumerics, with dashes (-), underscores (_), dots (.), and
      alphanumerics between.
    bareMetalVersion: Required. The Anthos clusters on bare metal version for
      your user cluster.
    binaryAuthorization: Binary Authorization related configurations.
    clusterOperations: Cluster operations configuration.
    controlPlane: Required. Control plane configuration.
    createTime: Output only. The time when the bare metal user cluster was
      created.
    deleteTime: Output only. The time when the bare metal user cluster was
      deleted. If the resource is not deleted, this must be empty
    description: A human readable description of this bare metal user cluster.
    endpoint: Output only. The IP address of the bare metal user cluster's API
      server.
    etag: Output only. This checksum is computed by the server based on the
      value of other fields, and may be sent on update and delete requests to
      ensure the client has an up-to-date value before proceeding. Allows
      clients to perform consistent read-modify-writes through optimistic
      concurrency control.
    fleet: Output only. Fleet configuration for the cluster.
    loadBalancer: Required. Load balancer configuration.
    localName: Output only. The object name of the bare metal user cluster
      custom resource on the associated admin cluster. This field is used to
      support conflicting names when enrolling existing clusters to the API.
      When used as a part of cluster enrollment, this field will differ from
      the name in the resource name. For new clusters, this field will match
      the user provided cluster name and be visible in the last component of
      the resource name. It is not modifiable. When the local name and cluster
      name differ, the local name is used in the admin cluster controller
      logs. You use the cluster name when accessing the cluster using bmctl
      and kubectl.
    maintenanceConfig: Maintenance configuration.
    maintenanceStatus: Output only. Status of on-going maintenance tasks.
    name: Immutable. The bare metal user cluster resource name.
    networkConfig: Required. Network configuration.
    nodeAccessConfig: Node access related configurations.
    nodeConfig: Workload node configuration.
    osEnvironmentConfig: OS environment related configurations.
    proxy: Proxy configuration.
    reconciling: Output only. If set, there are currently changes in flight to
      the bare metal user cluster.
    securityConfig: Security related setting configuration.
    state: Output only. The current state of the bare metal user cluster.
    status: Output only. Detailed cluster status.
    storage: Required. Storage configuration.
    uid: Output only. The unique identifier of the bare metal user cluster.
    updateTime: Output only. The time when the bare metal user cluster was
      last updated.
    upgradePolicy: The cluster upgrade policy.
    validationCheck: Output only. The result of the preflight check.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. The current state of the bare metal user cluster.

    Values:
      STATE_UNSPECIFIED: Not set.
      PROVISIONING: The PROVISIONING state indicates the cluster is being
        created.
      RUNNING: The RUNNING state indicates the cluster has been created and is
        fully usable.
      RECONCILING: The RECONCILING state indicates that the cluster is being
        updated. It remains available, but potentially with degraded
        performance.
      STOPPING: The STOPPING state indicates the cluster is being deleted.
      ERROR: The ERROR state indicates the cluster is in a broken
        unrecoverable state.
      DEGRADED: The DEGRADED state indicates the cluster requires user action
        to restore full functionality.
    """
        STATE_UNSPECIFIED = 0
        PROVISIONING = 1
        RUNNING = 2
        RECONCILING = 3
        STOPPING = 4
        ERROR = 5
        DEGRADED = 6

    @encoding.MapUnrecognizedFields('additionalProperties')
    class AnnotationsValue(_messages.Message):
        """Annotations on the bare metal user cluster. This field has the same
    restrictions as Kubernetes annotations. The total size of all keys and
    values combined is limited to 256k. Key can have 2 segments: prefix
    (optional) and name (required), separated by a slash (/). Prefix must be a
    DNS subdomain. Name must be 63 characters or less, begin and end with
    alphanumerics, with dashes (-), underscores (_), dots (.), and
    alphanumerics between.

    Messages:
      AdditionalProperty: An additional property for a AnnotationsValue
        object.

    Fields:
      additionalProperties: Additional properties of type AnnotationsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a AnnotationsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    adminClusterMembership = _messages.StringField(1)
    adminClusterName = _messages.StringField(2)
    annotations = _messages.MessageField('AnnotationsValue', 3)
    bareMetalVersion = _messages.StringField(4)
    binaryAuthorization = _messages.MessageField('BinaryAuthorization', 5)
    clusterOperations = _messages.MessageField('BareMetalClusterOperationsConfig', 6)
    controlPlane = _messages.MessageField('BareMetalControlPlaneConfig', 7)
    createTime = _messages.StringField(8)
    deleteTime = _messages.StringField(9)
    description = _messages.StringField(10)
    endpoint = _messages.StringField(11)
    etag = _messages.StringField(12)
    fleet = _messages.MessageField('Fleet', 13)
    loadBalancer = _messages.MessageField('BareMetalLoadBalancerConfig', 14)
    localName = _messages.StringField(15)
    maintenanceConfig = _messages.MessageField('BareMetalMaintenanceConfig', 16)
    maintenanceStatus = _messages.MessageField('BareMetalMaintenanceStatus', 17)
    name = _messages.StringField(18)
    networkConfig = _messages.MessageField('BareMetalNetworkConfig', 19)
    nodeAccessConfig = _messages.MessageField('BareMetalNodeAccessConfig', 20)
    nodeConfig = _messages.MessageField('BareMetalWorkloadNodeConfig', 21)
    osEnvironmentConfig = _messages.MessageField('BareMetalOsEnvironmentConfig', 22)
    proxy = _messages.MessageField('BareMetalProxyConfig', 23)
    reconciling = _messages.BooleanField(24)
    securityConfig = _messages.MessageField('BareMetalSecurityConfig', 25)
    state = _messages.EnumField('StateValueValuesEnum', 26)
    status = _messages.MessageField('ResourceStatus', 27)
    storage = _messages.MessageField('BareMetalStorageConfig', 28)
    uid = _messages.StringField(29)
    updateTime = _messages.StringField(30)
    upgradePolicy = _messages.MessageField('BareMetalClusterUpgradePolicy', 31)
    validationCheck = _messages.MessageField('ValidationCheck', 32)