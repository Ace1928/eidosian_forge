from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MessagePublishingRoute(_messages.Message):
    """MessagePublishingRoute is a resource for publishing messages to.

  Messages:
    LabelsValue: Optional. Labels of the resource.

  Fields:
    createTime: Output only. The timestamp when the resource was created.
    description: Optional. A free-text description of the resource. Max length
      1024 characters.
    gateways: Required. Gateways defines a list of gateways this
      MessagePublishRoute is attached to, as one of the routes to publish
      messages to. Each gateway reference should match the pattern:
      `projects/*/locations/*/gateways/`
    labels: Optional. Labels of the resource.
    name: Identifier. Name of the MessagePublishRoute resource. It matches
      pattern `projects/*/locations/*/messagePublishingRoutes/`.
    updateTime: Output only. The timestamp when the resource was updated.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Optional. Labels of the resource.

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
    description = _messages.StringField(2)
    gateways = _messages.StringField(3, repeated=True)
    labels = _messages.MessageField('LabelsValue', 4)
    name = _messages.StringField(5)
    updateTime = _messages.StringField(6)