from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OrchestrationCluster(_messages.Message):
    """Orchestration cluster represents a GKE cluster with config controller
  and TNA specific components installed on it.

  Enums:
    StateValueValuesEnum: Output only. State of the Orchestration Cluster.

  Messages:
    LabelsValue: Labels as key value pairs.

  Fields:
    createTime: Output only. [Output only] Create time stamp.
    labels: Labels as key value pairs.
    managementConfig: Management configuration of the underlying GKE cluster.
    name: Name of the orchestration cluster. The name of orchestration cluster
      cannot be more than 24 characters.
    state: Output only. State of the Orchestration Cluster.
    tnaVersion: Output only. Provides the TNA version installed on the
      cluster.
    updateTime: Output only. [Output only] Update time stamp.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. State of the Orchestration Cluster.

    Values:
      STATE_UNSPECIFIED: Unspecified state.
      CREATING: OrchestrationCluster is being created.
      ACTIVE: OrchestrationCluster has been created and is ready for use.
      DELETING: OrchestrationCluster is being deleted.
      FAILED: OrchestrationCluster encountered an error and is in an
        indeterministic state. User can still initiate a delete operation on
        this state.
    """
        STATE_UNSPECIFIED = 0
        CREATING = 1
        ACTIVE = 2
        DELETING = 3
        FAILED = 4

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Labels as key value pairs.

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
    createTime = _messages.StringField(1)
    labels = _messages.MessageField('LabelsValue', 2)
    managementConfig = _messages.MessageField('ManagementConfig', 3)
    name = _messages.StringField(4)
    state = _messages.EnumField('StateValueValuesEnum', 5)
    tnaVersion = _messages.StringField(6)
    updateTime = _messages.StringField(7)