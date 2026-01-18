from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TriggerEntity(_messages.Message):
    """Trigger is not used as an independent entity, it is retrieved as part of
  a Table entity.

  Messages:
    CustomFeaturesValue: Custom engine specific features.

  Fields:
    customFeatures: Custom engine specific features.
    name: The name of the trigger.
    sqlCode: The SQL code which creates the trigger.
    triggerType: Indicates when the trigger fires, for example BEFORE
      STATEMENT, AFTER EACH ROW.
    triggeringEvents: The DML, DDL, or database events that fire the trigger,
      for example INSERT, UPDATE.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class CustomFeaturesValue(_messages.Message):
        """Custom engine specific features.

    Messages:
      AdditionalProperty: An additional property for a CustomFeaturesValue
        object.

    Fields:
      additionalProperties: Properties of the object.
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a CustomFeaturesValue object.

      Fields:
        key: Name of the additional property.
        value: A extra_types.JsonValue attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('extra_types.JsonValue', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    customFeatures = _messages.MessageField('CustomFeaturesValue', 1)
    name = _messages.StringField(2)
    sqlCode = _messages.StringField(3)
    triggerType = _messages.StringField(4)
    triggeringEvents = _messages.StringField(5, repeated=True)