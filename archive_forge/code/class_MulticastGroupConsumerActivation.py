from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MulticastGroupConsumerActivation(_messages.Message):
    """Multicast group consumer activation resource.

  Messages:
    LabelsValue: Optional. Labels as key-value pairs

  Fields:
    createTime: Output only. [Output only] The timestamp when the multicast
      group consumer activation was created.
    labels: Optional. Labels as key-value pairs
    multicastConsumerAssociation: Required. The resource name of the multicast
      consumer association that is in the same zone as this multicast group
      consumer activation. Use the following format:
      `projects/*/locations/*/multicastConsumerAssociations/*`.
    multicastGroup: Required. The resource name of the multicast group created
      by the producer in the same zone as this multicast group consumer
      activation. Use the following format: //
      `projects/*/locations/*/multicastGroups/*`.
    name: Identifier. The resource name of the multicast group consumer
      activation. Use the following format:
      `projects/*/locations/*/multicastGroupConsumerActivations/*`.
    updateTime: Output only. [Output only] The timestamp when the multicast
      group consumer activation was most recently updated.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Optional. Labels as key-value pairs

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
    multicastConsumerAssociation = _messages.StringField(3)
    multicastGroup = _messages.StringField(4)
    name = _messages.StringField(5)
    updateTime = _messages.StringField(6)