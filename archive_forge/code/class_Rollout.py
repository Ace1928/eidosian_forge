from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Rollout(_messages.Message):
    """Rollout contains the Rollout metadata and configuration.

  Enums:
    StateValueValuesEnum: Output only. State specifies various states of the
      Rollout.

  Messages:
    AnnotationsValue: Optional. Annotations for this Rollout.
    LabelsValue: Optional. Labels for this Rollout.

  Fields:
    annotations: Optional. Annotations for this Rollout.
    clusterStatus: Output only. Metadata about the cluster status which are
      part of the Rollout. Provided by the server.
    completeTime: Output only. The timestamp at which the Rollout was
      completed.
    createTime: Output only. The timestamp at which the Rollout was created.
    deleteTime: Output only. The timestamp at the Rollout was deleted.
    displayName: Optional. Human readable display name of the Rollout.
    etag: Output only. etag of the Rollout Ex. abc1234
    feature: Optional. Feature config to use for Rollout.
    labels: Optional. Labels for this Rollout.
    managedRolloutConfig: Optional. The configuration used for the Rollout.
    name: Identifier. The full, unique resource name of this Rollout in the
      format of `projects/{project}/locations/global/rollouts/{rollout}`.
    state: Output only. State specifies various states of the Rollout.
    uid: Output only. Google-generated UUID for this resource. This is unique
      across all Rollout resources. If a Rollout resource is deleted and
      another resource with the same name is created, it gets a different uid.
    updateTime: Output only. The timestamp at which the Rollout was last
      updated.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. State specifies various states of the Rollout.

    Values:
      STATE_UNSPECIFIED: Unspecified state.
      RUNNING: The Rollout is running.
      PAUSED: The Rollout is paused.
      CANCELLED: The Rollout is in a failure terminal state.
      COMPLETED: The Rollout is in a terminal state.
    """
        STATE_UNSPECIFIED = 0
        RUNNING = 1
        PAUSED = 2
        CANCELLED = 3
        COMPLETED = 4

    @encoding.MapUnrecognizedFields('additionalProperties')
    class AnnotationsValue(_messages.Message):
        """Optional. Annotations for this Rollout.

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

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Optional. Labels for this Rollout.

    Messages:
      AdditionalProperty: An additional property for a LabelsValue object.

    Fields:
      additionalProperties: Additional properties of type LabelsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a LabelsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    annotations = _messages.MessageField('AnnotationsValue', 1)
    clusterStatus = _messages.MessageField('ClusterStatus', 2, repeated=True)
    completeTime = _messages.StringField(3)
    createTime = _messages.StringField(4)
    deleteTime = _messages.StringField(5)
    displayName = _messages.StringField(6)
    etag = _messages.StringField(7)
    feature = _messages.MessageField('FeatureUpdate', 8)
    labels = _messages.MessageField('LabelsValue', 9)
    managedRolloutConfig = _messages.MessageField('ManagedRolloutConfig', 10)
    name = _messages.StringField(11)
    state = _messages.EnumField('StateValueValuesEnum', 12)
    uid = _messages.StringField(13)
    updateTime = _messages.StringField(14)