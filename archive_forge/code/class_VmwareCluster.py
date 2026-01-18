from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VmwareCluster(_messages.Message):
    """Resource that represents a VMware user cluster. ##

  Enums:
    StateValueValuesEnum: Output only. The current state of VMware user
      cluster.

  Messages:
    AnnotationsValue: Annotations on the VMware user cluster. This field has
      the same restrictions as Kubernetes annotations. The total size of all
      keys and values combined is limited to 256k. Key can have 2 segments:
      prefix (optional) and name (required), separated by a slash (/). Prefix
      must be a DNS subdomain. Name must be 63 characters or less, begin and
      end with alphanumerics, with dashes (-), underscores (_), dots (.), and
      alphanumerics between.

  Fields:
    adminClusterMembership: Required. The admin cluster this VMware user
      cluster belongs to. This is the full resource name of the admin
      cluster's fleet membership. In the future, references to other resource
      types might be allowed if admin clusters are modeled as their own
      resources.
    adminClusterName: Output only. The resource name of the VMware admin
      cluster hosting this user cluster.
    annotations: Annotations on the VMware user cluster. This field has the
      same restrictions as Kubernetes annotations. The total size of all keys
      and values combined is limited to 256k. Key can have 2 segments: prefix
      (optional) and name (required), separated by a slash (/). Prefix must be
      a DNS subdomain. Name must be 63 characters or less, begin and end with
      alphanumerics, with dashes (-), underscores (_), dots (.), and
      alphanumerics between.
    antiAffinityGroups: AAGConfig specifies whether to spread VMware user
      cluster nodes across at least three physical hosts in the datacenter.
    authorization: RBAC policy that will be applied and managed by the Anthos
      On-Prem API.
    autoRepairConfig: Configuration for auto repairing.
    binaryAuthorization: Binary Authorization related configurations.
    controlPlaneNode: VMware user cluster control plane nodes must have either
      1 or 3 replicas.
    createTime: Output only. The time at which VMware user cluster was
      created.
    dataplaneV2: VmwareDataplaneV2Config specifies configuration for Dataplane
      V2.
    deleteTime: Output only. The time at which VMware user cluster was
      deleted.
    description: A human readable description of this VMware user cluster.
    disableBundledIngress: Disable bundled ingress.
    enableControlPlaneV2: Enable control plane V2. Default to false.
    endpoint: Output only. The DNS name of VMware user cluster's API server.
    etag: This checksum is computed by the server based on the value of other
      fields, and may be sent on update and delete requests to ensure the
      client has an up-to-date value before proceeding. Allows clients to
      perform consistent read-modify-writes through optimistic concurrency
      control.
    fleet: Output only. Fleet configuration for the cluster.
    loadBalancer: Load balancer configuration.
    localName: Output only. The object name of the VMware OnPremUserCluster
      custom resource on the associated admin cluster. This field is used to
      support conflicting names when enrolling existing clusters to the API.
      When used as a part of cluster enrollment, this field will differ from
      the ID in the resource name. For new clusters, this field will match the
      user provided cluster name and be visible in the last component of the
      resource name. It is not modifiable. All users should use this name to
      access their cluster using gkectl or kubectl and should expect to see
      the local name when viewing admin cluster controller logs.
    name: Immutable. The VMware user cluster resource name.
    networkConfig: The VMware user cluster network configuration.
    onPremVersion: Required. The Anthos clusters on the VMware version for
      your user cluster.
    reconciling: Output only. If set, there are currently changes in flight to
      the VMware user cluster.
    state: Output only. The current state of VMware user cluster.
    status: Output only. ResourceStatus representing detailed cluster state.
    storage: Storage configuration.
    uid: Output only. The unique identifier of the VMware user cluster.
    updateTime: Output only. The time at which VMware user cluster was last
      updated.
    upgradePolicy: Specifies upgrade policy for the cluster.
    validationCheck: Output only. ValidationCheck represents the result of the
      preflight check job.
    vcenter: VmwareVCenterConfig specifies vCenter config for the user
      cluster. If unspecified, it is inherited from the admin cluster.
    vmTrackingEnabled: Enable VM tracking.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. The current state of VMware user cluster.

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
        """Annotations on the VMware user cluster. This field has the same
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
    antiAffinityGroups = _messages.MessageField('VmwareAAGConfig', 4)
    authorization = _messages.MessageField('Authorization', 5)
    autoRepairConfig = _messages.MessageField('VmwareAutoRepairConfig', 6)
    binaryAuthorization = _messages.MessageField('BinaryAuthorization', 7)
    controlPlaneNode = _messages.MessageField('VmwareControlPlaneNodeConfig', 8)
    createTime = _messages.StringField(9)
    dataplaneV2 = _messages.MessageField('VmwareDataplaneV2Config', 10)
    deleteTime = _messages.StringField(11)
    description = _messages.StringField(12)
    disableBundledIngress = _messages.BooleanField(13)
    enableControlPlaneV2 = _messages.BooleanField(14)
    endpoint = _messages.StringField(15)
    etag = _messages.StringField(16)
    fleet = _messages.MessageField('Fleet', 17)
    loadBalancer = _messages.MessageField('VmwareLoadBalancerConfig', 18)
    localName = _messages.StringField(19)
    name = _messages.StringField(20)
    networkConfig = _messages.MessageField('VmwareNetworkConfig', 21)
    onPremVersion = _messages.StringField(22)
    reconciling = _messages.BooleanField(23)
    state = _messages.EnumField('StateValueValuesEnum', 24)
    status = _messages.MessageField('ResourceStatus', 25)
    storage = _messages.MessageField('VmwareStorageConfig', 26)
    uid = _messages.StringField(27)
    updateTime = _messages.StringField(28)
    upgradePolicy = _messages.MessageField('VmwareClusterUpgradePolicy', 29)
    validationCheck = _messages.MessageField('ValidationCheck', 30)
    vcenter = _messages.MessageField('VmwareVCenterConfig', 31)
    vmTrackingEnabled = _messages.BooleanField(32)