from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VmwareNodePool(_messages.Message):
    """Resource VmwareNodePool represents a VMware node pool. ##

  Enums:
    StateValueValuesEnum: Output only. The current state of the node pool.

  Messages:
    AnnotationsValue: Annotations on the node pool. This field has the same
      restrictions as Kubernetes annotations. The total size of all keys and
      values combined is limited to 256k. Key can have 2 segments: prefix
      (optional) and name (required), separated by a slash (/). Prefix must be
      a DNS subdomain. Name must be 63 characters or less, begin and end with
      alphanumerics, with dashes (-), underscores (_), dots (.), and
      alphanumerics between.

  Fields:
    annotations: Annotations on the node pool. This field has the same
      restrictions as Kubernetes annotations. The total size of all keys and
      values combined is limited to 256k. Key can have 2 segments: prefix
      (optional) and name (required), separated by a slash (/). Prefix must be
      a DNS subdomain. Name must be 63 characters or less, begin and end with
      alphanumerics, with dashes (-), underscores (_), dots (.), and
      alphanumerics between.
    config: Required. The node configuration of the node pool.
    createTime: Output only. The time at which this node pool was created.
    deleteTime: Output only. The time at which this node pool was deleted. If
      the resource is not deleted, this must be empty
    displayName: The display name for the node pool.
    etag: This checksum is computed by the server based on the value of other
      fields, and may be sent on update and delete requests to ensure the
      client has an up-to-date value before proceeding. Allows clients to
      perform consistent read-modify-writes through optimistic concurrency
      control.
    name: Immutable. The resource name of this node pool.
    nodePoolAutoscaling: Node pool autoscaling config for the node pool.
    onPremVersion: Anthos version for the node pool. Defaults to the user
      cluster version.
    reconciling: Output only. If set, there are currently changes in flight to
      the node pool.
    state: Output only. The current state of the node pool.
    status: Output only. ResourceStatus representing the detailed VMware node
      pool state.
    uid: Output only. The unique identifier of the node pool.
    updateTime: Output only. The time at which this node pool was last
      updated.
    upgradePolicy: Upgrade policy for the node pool.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. The current state of the node pool.

    Values:
      STATE_UNSPECIFIED: Not set.
      PROVISIONING: The PROVISIONING state indicates the node pool is being
        created.
      RUNNING: The RUNNING state indicates the node pool has been created and
        is fully usable.
      RECONCILING: The RECONCILING state indicates that the node pool is being
        updated. It remains available, but potentially with degraded
        performance.
      STOPPING: The STOPPING state indicates the cluster is being deleted
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
        """Annotations on the node pool. This field has the same restrictions as
    Kubernetes annotations. The total size of all keys and values combined is
    limited to 256k. Key can have 2 segments: prefix (optional) and name
    (required), separated by a slash (/). Prefix must be a DNS subdomain. Name
    must be 63 characters or less, begin and end with alphanumerics, with
    dashes (-), underscores (_), dots (.), and alphanumerics between.

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
    config = _messages.MessageField('VmwareNodeConfig', 2)
    createTime = _messages.StringField(3)
    deleteTime = _messages.StringField(4)
    displayName = _messages.StringField(5)
    etag = _messages.StringField(6)
    name = _messages.StringField(7)
    nodePoolAutoscaling = _messages.MessageField('VmwareNodePoolAutoscalingConfig', 8)
    onPremVersion = _messages.StringField(9)
    reconciling = _messages.BooleanField(10)
    state = _messages.EnumField('StateValueValuesEnum', 11)
    status = _messages.MessageField('ResourceStatus', 12)
    uid = _messages.StringField(13)
    updateTime = _messages.StringField(14)
    upgradePolicy = _messages.MessageField('VmwareNodePoolUpgradePolicy', 15)