from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatacatalogV1ColumnSchemaLookerColumnSpec(_messages.Message):
    """Column info specific to Looker System.

  Enums:
    TypeValueValuesEnum: Looker specific column type of this column.

  Fields:
    type: Looker specific column type of this column.
  """

    class TypeValueValuesEnum(_messages.Enum):
        """Looker specific column type of this column.

    Values:
      LOOKER_COLUMN_TYPE_UNSPECIFIED: Unspecified.
      DIMENSION: Dimension.
      DIMENSION_GROUP: Dimension group - parent for Dimension.
      FILTER: Filter.
      MEASURE: Measure.
      PARAMETER: Parameter.
    """
        LOOKER_COLUMN_TYPE_UNSPECIFIED = 0
        DIMENSION = 1
        DIMENSION_GROUP = 2
        FILTER = 3
        MEASURE = 4
        PARAMETER = 5
    type = _messages.EnumField('TypeValueValuesEnum', 1)