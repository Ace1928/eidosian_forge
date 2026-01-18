from the Google Cloud Console. When you create a cluster with Anthos Multi-
from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudGkemulticloudV1AwsNodePool(_messages.Message):
    """An Anthos node pool running on AWS.

  Enums:
    StateValueValuesEnum: Output only. The lifecycle state of the node pool.

  Messages:
    AnnotationsValue: Optional. Annotations on the node pool. This field has
      the same restrictions as Kubernetes annotations. The total size of all
      keys and values combined is limited to 256k. Key can have 2 segments:
      prefix (optional) and name (required), separated by a slash (/). Prefix
      must be a DNS subdomain. Name must be 63 characters or less, begin and
      end with alphanumerics, with dashes (-), underscores (_), dots (.), and
      alphanumerics between.

  Fields:
    annotations: Optional. Annotations on the node pool. This field has the
      same restrictions as Kubernetes annotations. The total size of all keys
      and values combined is limited to 256k. Key can have 2 segments: prefix
      (optional) and name (required), separated by a slash (/). Prefix must be
      a DNS subdomain. Name must be 63 characters or less, begin and end with
      alphanumerics, with dashes (-), underscores (_), dots (.), and
      alphanumerics between.
    autoscaling: Required. Autoscaler configuration for this node pool.
    config: Required. The configuration of the node pool.
    createTime: Output only. The time at which this node pool was created.
    errors: Output only. A set of errors found in the node pool.
    etag: Allows clients to perform consistent read-modify-writes through
      optimistic concurrency control. Can be sent on update and delete
      requests to ensure the client has an up-to-date value before proceeding.
    management: Optional. The Management configuration for this node pool.
    maxPodsConstraint: Required. The constraint on the maximum number of pods
      that can be run simultaneously on a node in the node pool.
    name: The name of this resource. Node pool names are formatted as
      `projects//locations//awsClusters//awsNodePools/`. For more details on
      Google Cloud resource names, see [Resource
      Names](https://cloud.google.com/apis/design/resource_names)
    reconciling: Output only. If set, there are currently changes in flight to
      the node pool.
    state: Output only. The lifecycle state of the node pool.
    subnetId: Required. The subnet where the node pool node run.
    uid: Output only. A globally unique identifier for the node pool.
    updateSettings: Optional. Update settings control the speed and disruption
      of the update.
    updateTime: Output only. The time at which this node pool was last
      updated.
    version: Required. The Kubernetes version to run on this node pool (e.g.
      `1.19.10-gke.1000`). You can list all supported versions on a given
      Google Cloud region by calling GetAwsServerConfig.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. The lifecycle state of the node pool.

    Values:
      STATE_UNSPECIFIED: Not set.
      PROVISIONING: The PROVISIONING state indicates the node pool is being
        created.
      RUNNING: The RUNNING state indicates the node pool has been created and
        is fully usable.
      RECONCILING: The RECONCILING state indicates that the node pool is being
        reconciled.
      STOPPING: The STOPPING state indicates the node pool is being deleted.
      ERROR: The ERROR state indicates the node pool is in a broken
        unrecoverable state.
      DEGRADED: The DEGRADED state indicates the node pool requires user
        action to restore full functionality.
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
        """Optional. Annotations on the node pool. This field has the same
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
    autoscaling = _messages.MessageField('GoogleCloudGkemulticloudV1AwsNodePoolAutoscaling', 2)
    config = _messages.MessageField('GoogleCloudGkemulticloudV1AwsNodeConfig', 3)
    createTime = _messages.StringField(4)
    errors = _messages.MessageField('GoogleCloudGkemulticloudV1AwsNodePoolError', 5, repeated=True)
    etag = _messages.StringField(6)
    management = _messages.MessageField('GoogleCloudGkemulticloudV1AwsNodeManagement', 7)
    maxPodsConstraint = _messages.MessageField('GoogleCloudGkemulticloudV1MaxPodsConstraint', 8)
    name = _messages.StringField(9)
    reconciling = _messages.BooleanField(10)
    state = _messages.EnumField('StateValueValuesEnum', 11)
    subnetId = _messages.StringField(12)
    uid = _messages.StringField(13)
    updateSettings = _messages.MessageField('GoogleCloudGkemulticloudV1UpdateSettings', 14)
    updateTime = _messages.StringField(15)
    version = _messages.StringField(16)