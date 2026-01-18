from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MessageSubscription(_messages.Message):
    """MessageSubscription is the resource defining a subscription of a
  MessagePublishingRoute to deliver the message to.

  Messages:
    LabelsValue: Optional. Labels of the resource.

  Fields:
    createTime: Output only. The timestamp when the resource was created.
    description: Optional. A free-text description of the resource. Max length
      1024 characters.
    labels: Optional. Labels of the resource.
    messagePublishingRoute: Required. The MessagePublishingRoute resource name
      it's attached to. It matches pattern
      `projects/*/locations/*/messagePublishingRoutes/`.
    name: Identifier. Name of the MessageSubscription resource. It matches
      pattern `projects/*/locations/*/MessageSubscriptions/`.
    rules: Required. Rules that define how traffic is routed and handled. Each
      rule is matched independently. i.e. If all rules match, all the rules
      will take effect.
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
    labels = _messages.MessageField('LabelsValue', 3)
    messagePublishingRoute = _messages.StringField(4)
    name = _messages.StringField(5)
    rules = _messages.MessageField('MessageSubscriptionRouteRule', 6, repeated=True)
    updateTime = _messages.StringField(7)