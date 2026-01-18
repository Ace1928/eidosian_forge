from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MirroringDeployment(_messages.Message):
    """Message describing MirroringDeployment object

  Enums:
    StateValueValuesEnum: Output only. Current state of the deployment.

  Messages:
    LabelsValue: Optional. Labels as key value pairs

  Fields:
    createTime: Output only. [Output only] Create time stamp
    forwardingRule: Required. Immutable. The regional load balancer which the
      mirrored traffic should be forwarded to. Format is:
      projects/{project}/regions/{region}/forwardingRules/{forwardingRule}
    labels: Optional. Labels as key value pairs
    mirroringDeploymentGroup: Required. Immutable. The Mirroring Deployment
      Group that this resource is part of. Format is: `projects/{project}/loca
      tions/global/mirroringDeploymentGroups/{mirroringDeploymentGroup}`
    name: Immutable. Identifier. The name of the MirroringDeployment.
    reconciling: Output only. Whether reconciling is in progress, recommended
      per https://google.aip.dev/128.
    state: Output only. Current state of the deployment.
    updateTime: Output only. [Output only] Update time stamp
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. Current state of the deployment.

    Values:
      STATE_UNSPECIFIED: Not set.
      ACTIVE: Ready.
      CREATING: Being created.
      DELETING: Being deleted.
    """
        STATE_UNSPECIFIED = 0
        ACTIVE = 1
        CREATING = 2
        DELETING = 3

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Optional. Labels as key value pairs

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
    forwardingRule = _messages.StringField(2)
    labels = _messages.MessageField('LabelsValue', 3)
    mirroringDeploymentGroup = _messages.StringField(4)
    name = _messages.StringField(5)
    reconciling = _messages.BooleanField(6)
    state = _messages.EnumField('StateValueValuesEnum', 7)
    updateTime = _messages.StringField(8)