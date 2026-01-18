from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MulticastGroupDefinition(_messages.Message):
    """Multicast group definition resource.

  Messages:
    LabelsValue: Labels as key-value pairs.

  Fields:
    createTime: Output only. [Output only] The timestamp when the multicast
      group definition was created.
    ipCidrRange: Output only. [Output only] The multicast group IP address
      range.
    labels: Labels as key-value pairs.
    multicastDomain: Required. The resource name of the multicast domain in
      which to create this multicast group definition. Use the following
      format: `projects/*/locations/global/multicastDomains/*`.
    name: The resource name of the multicast group definition. Use the
      following format:
      `projects/*/locations/global/multicastGroupDefinitions/*`.
    reservedInternalRange: Required. The resource name of the internal range
      reserved for this multicast group definition. Use the following format:
      `projects/*/locations/global/internalRanges/*`.
    updateTime: Output only. [Output only] The timestamp whenthe multicast
      group definition was most recently updated.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Labels as key-value pairs.

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
    ipCidrRange = _messages.StringField(2)
    labels = _messages.MessageField('LabelsValue', 3)
    multicastDomain = _messages.StringField(4)
    name = _messages.StringField(5)
    reservedInternalRange = _messages.StringField(6)
    updateTime = _messages.StringField(7)