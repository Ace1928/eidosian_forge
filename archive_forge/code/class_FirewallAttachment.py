from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FirewallAttachment(_messages.Message):
    """Message describing Attachment object

  Enums:
    StateValueValuesEnum: Output only. Current state of the attachment.

  Messages:
    LabelsValue: Labels as key value pairs

  Fields:
    createTime: Output only. Create time stamp
    labels: Labels as key value pairs
    name: Immutable. Identifier. name of resource
    producerForwardingRuleName: Required. Name of the regional load balancer
      which the intercepted traffic should be forwarded to: 'projects/{project
      _id}/regions/{region}/forwardingRules/{forwardingRule}'
    producerNatSubnetworkName: Required. Name of the subnet that is used to
      NAT the IP addresses of the intercepted traffic in the attachment's VPC:
      'projects/{project_id}/{location}/subnetworks/{subnetwork_name}'
    reconciling: Output only. Whether reconciling is in progress, recommended
      per https://google.aip.dev/128.
    state: Output only. Current state of the attachment.
    updateTime: Output only. Update time stamp
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. Current state of the attachment.

    Values:
      STATE_UNSPECIFIED: Not set.
      CREATING: Being created.
      ACTIVE: Attachment is now active.
      DELETING: Being deleted.
      INACTIVE: Down or in an error state.
    """
        STATE_UNSPECIFIED = 0
        CREATING = 1
        ACTIVE = 2
        DELETING = 3
        INACTIVE = 4

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Labels as key value pairs

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
    name = _messages.StringField(3)
    producerForwardingRuleName = _messages.StringField(4)
    producerNatSubnetworkName = _messages.StringField(5)
    reconciling = _messages.BooleanField(6)
    state = _messages.EnumField('StateValueValuesEnum', 7)
    updateTime = _messages.StringField(8)