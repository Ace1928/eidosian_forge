from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatacatalogV1ColumnSchema(_messages.Message):
    """A column within a schema. Columns can be nested inside other columns.

  Enums:
    HighestIndexingTypeValueValuesEnum: Optional. Most important inclusion of
      this column.

  Fields:
    column: Required. Name of the column. Must be a UTF-8 string without dots
      (.). The maximum size is 64 bytes.
    defaultValue: Optional. Default value for the column.
    description: Optional. Description of the column. Default value is an
      empty string. The description must be a UTF-8 string with the maximum
      size of 2000 bytes.
    gcRule: Optional. Garbage collection policy for the column or column
      family. Applies to systems like Cloud Bigtable.
    highestIndexingType: Optional. Most important inclusion of this column.
    lookerColumnSpec: Looker specific column info of this column.
    mode: Optional. A column's mode indicates whether values in this column
      are required, nullable, or repeated. Only `NULLABLE`, `REQUIRED`, and
      `REPEATED` values are supported. Default mode is `NULLABLE`.
    ordinalPosition: Optional. Ordinal position
    rangeElementType: Optional. The subtype of the RANGE, if the type of this
      field is RANGE. If the type is RANGE, this field is required. Possible
      values for the field element type of a RANGE include: * DATE * DATETIME
      * TIMESTAMP
    subcolumns: Optional. Schema of sub-columns. A column can have zero or
      more sub-columns.
    type: Required. Type of the column. Must be a UTF-8 string with the
      maximum size of 128 bytes.
  """

    class HighestIndexingTypeValueValuesEnum(_messages.Enum):
        """Optional. Most important inclusion of this column.

    Values:
      INDEXING_TYPE_UNSPECIFIED: Unspecified.
      INDEXING_TYPE_NONE: Column not a part of an index.
      INDEXING_TYPE_NON_UNIQUE: Column Part of non unique index.
      INDEXING_TYPE_UNIQUE: Column part of unique index.
      INDEXING_TYPE_PRIMARY_KEY: Column part of the primary key.
    """
        INDEXING_TYPE_UNSPECIFIED = 0
        INDEXING_TYPE_NONE = 1
        INDEXING_TYPE_NON_UNIQUE = 2
        INDEXING_TYPE_UNIQUE = 3
        INDEXING_TYPE_PRIMARY_KEY = 4
    column = _messages.StringField(1)
    defaultValue = _messages.StringField(2)
    description = _messages.StringField(3)
    gcRule = _messages.StringField(4)
    highestIndexingType = _messages.EnumField('HighestIndexingTypeValueValuesEnum', 5)
    lookerColumnSpec = _messages.MessageField('GoogleCloudDatacatalogV1ColumnSchemaLookerColumnSpec', 6)
    mode = _messages.StringField(7)
    ordinalPosition = _messages.IntegerField(8, variant=_messages.Variant.INT32)
    rangeElementType = _messages.MessageField('GoogleCloudDatacatalogV1ColumnSchemaFieldElementType', 9)
    subcolumns = _messages.MessageField('GoogleCloudDatacatalogV1ColumnSchema', 10, repeated=True)
    type = _messages.StringField(11)