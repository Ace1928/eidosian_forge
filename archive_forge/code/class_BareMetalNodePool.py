from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BareMetalNodePool(_messages.Message):
    """Resource that represents a bare metal node pool.

  Enums:
    StateValueValuesEnum: Output only. The current state of the bare metal
      node pool.

  Messages:
    AnnotationsValue: Annotations on the bare metal node pool. This field has
      the same restrictions as Kubernetes annotations. The total size of all
      keys and values combined is limited to 256k. Key can have 2 segments:
      prefix (optional) and name (required), separated by a slash (/). Prefix
      must be a DNS subdomain. Name must be 63 characters or less, begin and
      end with alphanumerics, with dashes (-), underscores (_), dots (.), and
      alphanumerics between.

  Fields:
    annotations: Annotations on the bare metal node pool. This field has the
      same restrictions as Kubernetes annotations. The total size of all keys
      and values combined is limited to 256k. Key can have 2 segments: prefix
      (optional) and name (required), separated by a slash (/). Prefix must be
      a DNS subdomain. Name must be 63 characters or less, begin and end with
      alphanumerics, with dashes (-), underscores (_), dots (.), and
      alphanumerics between.
    bareMetalVersion: Specifies node pool version. The field is used to
      upgrade the nodepool to the specified version. When specified during
      node pool creation, the maximum allowed version skew between cluster and
      nodepool is 1 minor version. When the field is not specified during
      nodepool creation, the nodepool is created at the cluster version.
    createTime: Output only. The time at which this bare metal node pool was
      created.
    deleteTime: Output only. The time at which this bare metal node pool was
      deleted. If the resource is not deleted, this must be empty
    displayName: The display name for the bare metal node pool.
    etag: This checksum is computed by the server based on the value of other
      fields, and may be sent on update and delete requests to ensure the
      client has an up-to-date value before proceeding. Allows clients to
      perform consistent read-modify-writes through optimistic concurrency
      control.
    name: Immutable. The bare metal node pool resource name.
    nodePoolConfig: Required. Node pool configuration.
    reconciling: Output only. If set, there are currently changes in flight to
      the bare metal node pool.
    state: Output only. The current state of the bare metal node pool.
    status: Output only. ResourceStatus representing the detailed node pool
      status.
    uid: Output only. The unique identifier of the bare metal node pool.
    updateTime: Output only. The time at which this bare metal node pool was
      last updated.
    upgradePolicy: The worker node pool upgrade policy.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. The current state of the bare metal node pool.

    Values:
      STATE_UNSPECIFIED: Not set.
      PROVISIONING: The PROVISIONING state indicates the bare metal node pool
        is being created.
      RUNNING: The RUNNING state indicates the bare metal node pool has been
        created and is fully usable.
      RECONCILING: The RECONCILING state indicates that the bare metal node
        pool is being updated. It remains available, but potentially with
        degraded performance.
      STOPPING: The STOPPING state indicates the bare metal node pool is being
        deleted.
      ERROR: The ERROR state indicates the bare metal node pool is in a broken
        unrecoverable state.
      DEGRADED: The DEGRADED state indicates the bare metal node pool requires
        user action to restore full functionality.
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
        """Annotations on the bare metal node pool. This field has the same
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
    createTime = _messages.StringField(3)
    deleteTime = _messages.StringField(4)
    displayName = _messages.StringField(5)
    etag = _messages.StringField(6)
    name = _messages.StringField(7)
    nodePoolConfig = _messages.MessageField('BareMetalNodePoolConfig', 8)
    reconciling = _messages.BooleanField(9)
    state = _messages.EnumField('StateValueValuesEnum', 10)
    status = _messages.MessageField('ResourceStatus', 11)
    uid = _messages.StringField(12)
    updateTime = _messages.StringField(13)
    upgradePolicy = _messages.MessageField('BareMetalNodePoolUpgradePolicy', 14)