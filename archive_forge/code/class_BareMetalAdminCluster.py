from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BareMetalAdminCluster(_messages.Message):
    """Resource that represents a bare metal admin cluster.

  Enums:
    StateValueValuesEnum: Output only. The current state of the bare metal
      admin cluster.

  Messages:
    AnnotationsValue: Annotations on the bare metal admin cluster. This field
      has the same restrictions as Kubernetes annotations. The total size of
      all keys and values combined is limited to 256k. Key can have 2
      segments: prefix (optional) and name (required), separated by a slash
      (/). Prefix must be a DNS subdomain. Name must be 63 characters or less,
      begin and end with alphanumerics, with dashes (-), underscores (_), dots
      (.), and alphanumerics between.

  Fields:
    annotations: Annotations on the bare metal admin cluster. This field has
      the same restrictions as Kubernetes annotations. The total size of all
      keys and values combined is limited to 256k. Key can have 2 segments:
      prefix (optional) and name (required), separated by a slash (/). Prefix
      must be a DNS subdomain. Name must be 63 characters or less, begin and
      end with alphanumerics, with dashes (-), underscores (_), dots (.), and
      alphanumerics between.
    bareMetalVersion: The Anthos clusters on bare metal version for the bare
      metal admin cluster.
    binaryAuthorization: Binary Authorization related configurations.
    clusterOperations: Cluster operations configuration.
    controlPlane: Control plane configuration.
    createTime: Output only. The time at which this bare metal admin cluster
      was created.
    deleteTime: Output only. The time at which this bare metal admin cluster
      was deleted. If the resource is not deleted, this must be empty
    description: A human readable description of this bare metal admin
      cluster.
    endpoint: Output only. The IP address name of bare metal admin cluster's
      API server.
    etag: This checksum is computed by the server based on the value of other
      fields, and may be sent on update and delete requests to ensure the
      client has an up-to-date value before proceeding. Allows clients to
      perform consistent read-modify-writes through optimistic concurrency
      control.
    fleet: Output only. Fleet configuration for the cluster.
    loadBalancer: Load balancer configuration.
    localName: Output only. The object name of the bare metal cluster custom
      resource. This field is used to support conflicting names when enrolling
      existing clusters to the API. When used as a part of cluster enrollment,
      this field will differ from the ID in the resource name. For new
      clusters, this field will match the user provided cluster name and be
      visible in the last component of the resource name. It is not
      modifiable. All users should use this name to access their cluster using
      gkectl or kubectl and should expect to see the local name when viewing
      admin cluster controller logs.
    maintenanceConfig: Maintenance configuration.
    maintenanceStatus: Output only. MaintenanceStatus representing state of
      maintenance.
    name: Immutable. The bare metal admin cluster resource name.
    networkConfig: Network configuration.
    nodeAccessConfig: Node access related configurations.
    nodeConfig: Workload node configuration.
    osEnvironmentConfig: OS environment related configurations.
    proxy: Proxy configuration.
    reconciling: Output only. If set, there are currently changes in flight to
      the bare metal Admin Cluster.
    securityConfig: Security related configuration.
    state: Output only. The current state of the bare metal admin cluster.
    status: Output only. ResourceStatus representing detailed cluster status.
    storage: Storage configuration.
    uid: Output only. The unique identifier of the bare metal admin cluster.
    updateTime: Output only. The time at which this bare metal admin cluster
      was last updated.
    validationCheck: Output only. ValidationCheck representing the result of
      the preflight check.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. The current state of the bare metal admin cluster.

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
        """Annotations on the bare metal admin cluster. This field has the same
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
    annotations = _messages.MessageField('AnnotationsValue', 1)
    bareMetalVersion = _messages.StringField(2)
    binaryAuthorization = _messages.MessageField('BinaryAuthorization', 3)
    clusterOperations = _messages.MessageField('BareMetalAdminClusterOperationsConfig', 4)
    controlPlane = _messages.MessageField('BareMetalAdminControlPlaneConfig', 5)
    createTime = _messages.StringField(6)
    deleteTime = _messages.StringField(7)
    description = _messages.StringField(8)
    endpoint = _messages.StringField(9)
    etag = _messages.StringField(10)
    fleet = _messages.MessageField('Fleet', 11)
    loadBalancer = _messages.MessageField('BareMetalAdminLoadBalancerConfig', 12)
    localName = _messages.StringField(13)
    maintenanceConfig = _messages.MessageField('BareMetalAdminMaintenanceConfig', 14)
    maintenanceStatus = _messages.MessageField('BareMetalAdminMaintenanceStatus', 15)
    name = _messages.StringField(16)
    networkConfig = _messages.MessageField('BareMetalAdminNetworkConfig', 17)
    nodeAccessConfig = _messages.MessageField('BareMetalAdminNodeAccessConfig', 18)
    nodeConfig = _messages.MessageField('BareMetalAdminWorkloadNodeConfig', 19)
    osEnvironmentConfig = _messages.MessageField('BareMetalAdminOsEnvironmentConfig', 20)
    proxy = _messages.MessageField('BareMetalAdminProxyConfig', 21)
    reconciling = _messages.BooleanField(22)
    securityConfig = _messages.MessageField('BareMetalAdminSecurityConfig', 23)
    state = _messages.EnumField('StateValueValuesEnum', 24)
    status = _messages.MessageField('ResourceStatus', 25)
    storage = _messages.MessageField('BareMetalAdminStorageConfig', 26)
    uid = _messages.StringField(27)
    updateTime = _messages.StringField(28)
    validationCheck = _messages.MessageField('ValidationCheck', 29)