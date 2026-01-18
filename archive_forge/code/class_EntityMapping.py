from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EntityMapping(_messages.Message):
    """Details of the mappings of a database entity.

  Enums:
    DraftTypeValueValuesEnum: Type of draft entity.
    SourceTypeValueValuesEnum: Type of source entity.

  Fields:
    draftEntity: Target entity full name. The draft entity can also include a
      column, index or constraint using the same naming notation
      schema.table.column.
    draftType: Type of draft entity.
    mappingLog: Entity mapping log entries. Multiple rules can be effective
      and contribute changes to a converted entity, such as a rule can handle
      the entity name, another rule can handle an entity type. In addition,
      rules which did not change the entity are also logged along with the
      reason preventing them to do so.
    sourceEntity: Source entity full name. The source entity can also be a
      column, index or constraint using the same naming notation
      schema.table.column.
    sourceType: Type of source entity.
  """

    class DraftTypeValueValuesEnum(_messages.Enum):
        """Type of draft entity.

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

    class SourceTypeValueValuesEnum(_messages.Enum):
        """Type of source entity.

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
    draftEntity = _messages.StringField(1)
    draftType = _messages.EnumField('DraftTypeValueValuesEnum', 2)
    mappingLog = _messages.MessageField('EntityMappingLogEntry', 3, repeated=True)
    sourceEntity = _messages.StringField(4)
    sourceType = _messages.EnumField('SourceTypeValueValuesEnum', 5)