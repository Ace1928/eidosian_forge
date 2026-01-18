from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TableEntity(_messages.Message):
    """Table's parent is a schema.

  Messages:
    CustomFeaturesValue: Custom engine specific features.

  Fields:
    columns: Table columns.
    comment: Comment associated with the table.
    constraints: Table constraints.
    customFeatures: Custom engine specific features.
    indices: Table indices.
    triggers: Table triggers.
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
    columns = _messages.MessageField('ColumnEntity', 1, repeated=True)
    comment = _messages.StringField(2)
    constraints = _messages.MessageField('ConstraintEntity', 3, repeated=True)
    customFeatures = _messages.MessageField('CustomFeaturesValue', 4)
    indices = _messages.MessageField('IndexEntity', 5, repeated=True)
    triggers = _messages.MessageField('TriggerEntity', 6, repeated=True)