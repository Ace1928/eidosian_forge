from the Google Cloud Console. When you create a cluster with Anthos Multi-
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudGkemulticloudV1AttachedCluster(_messages.Message):
    """An Anthos cluster running on customer own infrastructure.

  Enums:
    StateValueValuesEnum: Output only. The current state of the cluster.

  Messages:
    AnnotationsValue: Optional. Annotations on the cluster. This field has the
      same restrictions as Kubernetes annotations. The total size of all keys
      and values combined is limited to 256k. Key can have 2 segments: prefix
      (optional) and name (required), separated by a slash (/). Prefix must be
      a DNS subdomain. Name must be 63 characters or less, begin and end with
      alphanumerics, with dashes (-), underscores (_), dots (.), and
      alphanumerics between.

  Fields:
    annotations: Optional. Annotations on the cluster. This field has the same
      restrictions as Kubernetes annotations. The total size of all keys and
      values combined is limited to 256k. Key can have 2 segments: prefix
      (optional) and name (required), separated by a slash (/). Prefix must be
      a DNS subdomain. Name must be 63 characters or less, begin and end with
      alphanumerics, with dashes (-), underscores (_), dots (.), and
      alphanumerics between.
    authorization: Optional. Configuration related to the cluster RBAC
      settings.
    binaryAuthorization: Optional. Binary Authorization configuration for this
      cluster.
    clusterRegion: Output only. The region where this cluster runs. For EKS
      clusters, this is a AWS region. For AKS clusters, this is an Azure
      region.
    createTime: Output only. The time at which this cluster was registered.
    description: Optional. A human readable description of this cluster.
      Cannot be longer than 255 UTF-8 encoded bytes.
    distribution: Required. The Kubernetes distribution of the underlying
      attached cluster. Supported values: ["eks", "aks", "generic"].
    errors: Output only. A set of errors found in the cluster.
    etag: Allows clients to perform consistent read-modify-writes through
      optimistic concurrency control. Can be sent on update and delete
      requests to ensure the client has an up-to-date value before proceeding.
    fleet: Required. Fleet configuration.
    kubernetesVersion: Output only. The Kubernetes version of the cluster.
    loggingConfig: Optional. Logging configuration for this cluster.
    monitoringConfig: Optional. Monitoring configuration for this cluster.
    name: The name of this resource. Cluster names are formatted as
      `projects//locations//attachedClusters/`. See [Resource
      Names](https://cloud.google.com/apis/design/resource_names) for more
      details on Google Cloud Platform resource names.
    oidcConfig: Required. OpenID Connect (OIDC) configuration for the cluster.
    platformVersion: Required. The platform version for the cluster (e.g.
      `1.19.0-gke.1000`). You can list all supported versions on a given
      Google Cloud region by calling GetAttachedServerConfig.
    proxyConfig: Optional. Proxy configuration for outbound HTTP(S) traffic.
    reconciling: Output only. If set, there are currently changes in flight to
      the cluster.
    state: Output only. The current state of the cluster.
    uid: Output only. A globally unique identifier for the cluster.
    updateTime: Output only. The time at which this cluster was last updated.
    workloadIdentityConfig: Output only. Workload Identity settings.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. The current state of the cluster.

    Values:
      STATE_UNSPECIFIED: Not set.
      PROVISIONING: The PROVISIONING state indicates the cluster is being
        registered.
      RUNNING: The RUNNING state indicates the cluster has been register and
        is fully usable.
      RECONCILING: The RECONCILING state indicates that some work is actively
        being done on the cluster, such as upgrading software components.
      STOPPING: The STOPPING state indicates the cluster is being de-
        registered.
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
        """Optional. Annotations on the cluster. This field has the same
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
    authorization = _messages.MessageField('GoogleCloudGkemulticloudV1AttachedClustersAuthorization', 2)
    binaryAuthorization = _messages.MessageField('GoogleCloudGkemulticloudV1BinaryAuthorization', 3)
    clusterRegion = _messages.StringField(4)
    createTime = _messages.StringField(5)
    description = _messages.StringField(6)
    distribution = _messages.StringField(7)
    errors = _messages.MessageField('GoogleCloudGkemulticloudV1AttachedClusterError', 8, repeated=True)
    etag = _messages.StringField(9)
    fleet = _messages.MessageField('GoogleCloudGkemulticloudV1Fleet', 10)
    kubernetesVersion = _messages.StringField(11)
    loggingConfig = _messages.MessageField('GoogleCloudGkemulticloudV1LoggingConfig', 12)
    monitoringConfig = _messages.MessageField('GoogleCloudGkemulticloudV1MonitoringConfig', 13)
    name = _messages.StringField(14)
    oidcConfig = _messages.MessageField('GoogleCloudGkemulticloudV1AttachedOidcConfig', 15)
    platformVersion = _messages.StringField(16)
    proxyConfig = _messages.MessageField('GoogleCloudGkemulticloudV1AttachedProxyConfig', 17)
    reconciling = _messages.BooleanField(18)
    state = _messages.EnumField('StateValueValuesEnum', 19)
    uid = _messages.StringField(20)
    updateTime = _messages.StringField(21)
    workloadIdentityConfig = _messages.MessageField('GoogleCloudGkemulticloudV1WorkloadIdentityConfig', 22)