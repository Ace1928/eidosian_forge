from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatacatalogV1DatabaseTableSpec(_messages.Message):
    """Specification that applies to a table resource. Valid only for entries
  with the `TABLE` type.

  Enums:
    TypeValueValuesEnum: Type of this table.

  Fields:
    databaseViewSpec: Spec what aplies to tables that are actually views. Not
      set for "real" tables.
    dataplexTable: Output only. Fields specific to a Dataplex table and
      present only in the Dataplex table entries.
    type: Type of this table.
  """

    class TypeValueValuesEnum(_messages.Enum):
        """Type of this table.

    Values:
      TABLE_TYPE_UNSPECIFIED: Default unknown table type.
      NATIVE: Native table.
      EXTERNAL: External table.
    """
        TABLE_TYPE_UNSPECIFIED = 0
        NATIVE = 1
        EXTERNAL = 2
    databaseViewSpec = _messages.MessageField('GoogleCloudDatacatalogV1DatabaseTableSpecDatabaseViewSpec', 1)
    dataplexTable = _messages.MessageField('GoogleCloudDatacatalogV1DataplexTableSpec', 2)
    type = _messages.EnumField('TypeValueValuesEnum', 3)