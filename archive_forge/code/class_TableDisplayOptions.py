from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TableDisplayOptions(_messages.Message):
    """Table display options that can be reused.

  Fields:
    shownColumns: Optional. This field is unused and has been replaced by
      TimeSeriesTable.column_settings
  """
    shownColumns = _messages.StringField(1, repeated=True)