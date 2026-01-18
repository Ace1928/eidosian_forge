from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MaintenanceExclusionWindow(_messages.Message):
    """Represents a maintenance exclusion window.

  Fields:
    id: Optional. A unique (per cluster) id for the window.
    window: Optional. The time window.
  """
    id = _messages.StringField(1)
    window = _messages.MessageField('TimeWindow', 2)