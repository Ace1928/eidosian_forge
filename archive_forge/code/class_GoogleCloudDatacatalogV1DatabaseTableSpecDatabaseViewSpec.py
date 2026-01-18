from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatacatalogV1DatabaseTableSpecDatabaseViewSpec(_messages.Message):
    """Specification that applies to database view.

  Enums:
    ViewTypeValueValuesEnum: Type of this view.

  Fields:
    baseTable: Name of a singular table this view reflects one to one.
    sqlQuery: SQL query used to generate this view.
    viewType: Type of this view.
  """

    class ViewTypeValueValuesEnum(_messages.Enum):
        """Type of this view.

    Values:
      VIEW_TYPE_UNSPECIFIED: Default unknown view type.
      STANDARD_VIEW: Standard view.
      MATERIALIZED_VIEW: Materialized view.
    """
        VIEW_TYPE_UNSPECIFIED = 0
        STANDARD_VIEW = 1
        MATERIALIZED_VIEW = 2
    baseTable = _messages.StringField(1)
    sqlQuery = _messages.StringField(2)
    viewType = _messages.EnumField('ViewTypeValueValuesEnum', 3)