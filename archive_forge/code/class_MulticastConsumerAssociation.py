from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MulticastConsumerAssociation(_messages.Message):
    """Multicast consumer association resource.

  Messages:
    LabelsValue: Labels as key-value pairs

  Fields:
    createTime: Output only. [Output only] The timestamp when the multicast
      consumer association was created.
    domainActivation: Reference to the domain activation in the same zone as
      the consumer association. [Deprecated] Use multicast_domain_activation
      instead.
    labels: Labels as key-value pairs
    multicastDomainActivation: Optional. The resource name of the multicast
      domain activation that is in the same zone as this multicast consumer
      association. Use the following format: //
      `projects/*/locations/*/multicastConsumerAssociations/*`.
    name: The resource name of the multicast consumer association. Use the
      following format:
      `projects/*/locations/*/multicastConsumerAssociations/*`.
    network: The resource name of the multicast consumer VPC network. Use
      following format:
      `projects/{project}/locations/global/networks/{network}`.
    updateTime: Output only. [Output only] The timestamp when the Multicast
      Consumer Association was most recently updated.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Labels as key-value pairs

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
    domainActivation = _messages.StringField(2)
    labels = _messages.MessageField('LabelsValue', 3)
    multicastDomainActivation = _messages.StringField(4)
    name = _messages.StringField(5)
    network = _messages.StringField(6)
    updateTime = _messages.StringField(7)