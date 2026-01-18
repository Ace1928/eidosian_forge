from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MappingRule(_messages.Message):
    """Definition of a transformation that is to be applied to a group of
  entities in the source schema. Several such transformations can be applied
  to an entity sequentially to define the corresponding entity in the target
  schema.

  Enums:
    RuleScopeValueValuesEnum: Required. The rule scope
    StateValueValuesEnum: Optional. The mapping rule state

  Fields:
    conditionalColumnSetValue: Optional. Rule to specify how the data
      contained in a column should be transformed (such as trimmed, rounded,
      etc) provided that the data meets certain criteria.
    convertRowidColumn: Optional. Rule to specify how multiple tables should
      be converted with an additional rowid column.
    displayName: Optional. A human readable name
    entityMove: Optional. Rule to specify how multiple entities should be
      relocated into a different schema.
    filter: Required. The rule filter
    filterTableColumns: Optional. Rule to specify the list of columns to
      include or exclude from a table.
    multiColumnDataTypeChange: Optional. Rule to specify how multiple columns
      should be converted to a different data type.
    multiEntityRename: Optional. Rule to specify how multiple entities should
      be renamed.
    name: Full name of the mapping rule resource, in the form of: projects/{pr
      oject}/locations/{location}/conversionWorkspaces/{set}/mappingRule/{rule
      }.
    revisionCreateTime: Output only. The timestamp that the revision was
      created.
    revisionId: Output only. The revision ID of the mapping rule. A new
      revision is committed whenever the mapping rule is changed in any way.
      The format is an 8-character hexadecimal string.
    ruleOrder: Required. The order in which the rule is applied. Lower order
      rules are applied before higher value rules so they may end up being
      overridden.
    ruleScope: Required. The rule scope
    setTablePrimaryKey: Optional. Rule to specify the primary key for a table
    singleColumnChange: Optional. Rule to specify how a single column is
      converted.
    singleEntityRename: Optional. Rule to specify how a single entity should
      be renamed.
    singlePackageChange: Optional. Rule to specify how a single package is
      converted.
    sourceSqlChange: Optional. Rule to change the sql code for an entity, for
      example, function, procedure.
    state: Optional. The mapping rule state
  """

    class RuleScopeValueValuesEnum(_messages.Enum):
        """Required. The rule scope

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

    class StateValueValuesEnum(_messages.Enum):
        """Optional. The mapping rule state

    Values:
      STATE_UNSPECIFIED: The state of the mapping rule is unknown.
      ENABLED: The rule is enabled.
      DISABLED: The rule is disabled.
      DELETED: The rule is logically deleted.
    """
        STATE_UNSPECIFIED = 0
        ENABLED = 1
        DISABLED = 2
        DELETED = 3
    conditionalColumnSetValue = _messages.MessageField('ConditionalColumnSetValue', 1)
    convertRowidColumn = _messages.MessageField('ConvertRowIdToColumn', 2)
    displayName = _messages.StringField(3)
    entityMove = _messages.MessageField('EntityMove', 4)
    filter = _messages.MessageField('MappingRuleFilter', 5)
    filterTableColumns = _messages.MessageField('FilterTableColumns', 6)
    multiColumnDataTypeChange = _messages.MessageField('MultiColumnDatatypeChange', 7)
    multiEntityRename = _messages.MessageField('MultiEntityRename', 8)
    name = _messages.StringField(9)
    revisionCreateTime = _messages.StringField(10)
    revisionId = _messages.StringField(11)
    ruleOrder = _messages.IntegerField(12)
    ruleScope = _messages.EnumField('RuleScopeValueValuesEnum', 13)
    setTablePrimaryKey = _messages.MessageField('SetTablePrimaryKey', 14)
    singleColumnChange = _messages.MessageField('SingleColumnChange', 15)
    singleEntityRename = _messages.MessageField('SingleEntityRename', 16)
    singlePackageChange = _messages.MessageField('SinglePackageChange', 17)
    sourceSqlChange = _messages.MessageField('SourceSqlChange', 18)
    state = _messages.EnumField('StateValueValuesEnum', 19)