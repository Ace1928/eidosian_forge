from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MaterializedViewStatus(_messages.Message):
    """Status of a materialized view. The last refresh timestamp status is
  omitted here, but is present in the MaterializedViewDefinition message.

  Fields:
    lastRefreshStatus: Output only. Error result of the last automatic
      refresh. If present, indicates that the last automatic refresh was
      unsuccessful.
    refreshWatermark: Output only. Refresh watermark of materialized view. The
      base tables' data were collected into the materialized view cache until
      this time.
  """
    lastRefreshStatus = _messages.MessageField('ErrorProto', 1)
    refreshWatermark = _messages.StringField(2)