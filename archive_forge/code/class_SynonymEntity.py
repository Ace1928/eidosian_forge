from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SynonymEntity(_messages.Message):
    """Synonym's parent is a schema.

  Enums:
    SourceTypeValueValuesEnum: The type of the entity for which the synonym is
      being created (usually a table or a sequence).

  Messages:
    CustomFeaturesValue: Custom engine specific features.

  Fields:
    customFeatures: Custom engine specific features.
    sourceEntity: The name of the entity for which the synonym is being
      created (the source).
    sourceType: The type of the entity for which the synonym is being created
      (usually a table or a sequence).
  """

    class SourceTypeValueValuesEnum(_messages.Enum):
        """The type of the entity for which the synonym is being created (usually
    a table or a sequence).

    Values:
      DATABASE_ENTITY_TYPE_UNSPECIFIED: Unspecified database entity type.
      DATABASE_ENTITY_TYPE_SCHEMA: Schema.
      DATABASE_ENTITY_TYPE_TABLE: Table.
      DATABASE_ENTITY_TYPE_COLUMN: Column.
      DATABASE_ENTITY_TYPE_CONSTRAINT: Constraint.
      DATABASE_ENTITY_TYPE_INDEX: Index.
      DATABASE_ENTITY_TYPE_TRIGGER: Trigger.
      DATABASE_ENTITY_TYPE_VIEW: View.
      DATABASE_ENTITY_TYPE_SEQUENCE: Sequence.
      DATABASE_ENTITY_TYPE_STORED_PROCEDURE: Stored Procedure.
      DATABASE_ENTITY_TYPE_FUNCTION: Function.
      DATABASE_ENTITY_TYPE_SYNONYM: Synonym.
      DATABASE_ENTITY_TYPE_DATABASE_PACKAGE: Package.
      DATABASE_ENTITY_TYPE_UDT: UDT.
      DATABASE_ENTITY_TYPE_MATERIALIZED_VIEW: Materialized View.
      DATABASE_ENTITY_TYPE_DATABASE: Database.
    """
        DATABASE_ENTITY_TYPE_UNSPECIFIED = 0
        DATABASE_ENTITY_TYPE_SCHEMA = 1
        DATABASE_ENTITY_TYPE_TABLE = 2
        DATABASE_ENTITY_TYPE_COLUMN = 3
        DATABASE_ENTITY_TYPE_CONSTRAINT = 4
        DATABASE_ENTITY_TYPE_INDEX = 5
        DATABASE_ENTITY_TYPE_TRIGGER = 6
        DATABASE_ENTITY_TYPE_VIEW = 7
        DATABASE_ENTITY_TYPE_SEQUENCE = 8
        DATABASE_ENTITY_TYPE_STORED_PROCEDURE = 9
        DATABASE_ENTITY_TYPE_FUNCTION = 10
        DATABASE_ENTITY_TYPE_SYNONYM = 11
        DATABASE_ENTITY_TYPE_DATABASE_PACKAGE = 12
        DATABASE_ENTITY_TYPE_UDT = 13
        DATABASE_ENTITY_TYPE_MATERIALIZED_VIEW = 14
        DATABASE_ENTITY_TYPE_DATABASE = 15

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
    sourceEntity = _messages.StringField(2)
    sourceType = _messages.EnumField('SourceTypeValueValuesEnum', 3)