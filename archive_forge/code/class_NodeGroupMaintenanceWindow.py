from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NodeGroupMaintenanceWindow(_messages.Message):
    """Time window specified for daily maintenance operations. GCE's internal
  maintenance will be performed within this window.

  Fields:
    maintenanceDuration: [Output only] A predetermined duration for the
      window, automatically chosen to be the smallest possible in the given
      scenario.
    startTime: Start time of the window. This must be in UTC format that
      resolves to one of 00:00, 04:00, 08:00, 12:00, 16:00, or 20:00. For
      example, both 13:00-5 and 08:00 are valid.
  """
    maintenanceDuration = _messages.MessageField('Duration', 1)
    startTime = _messages.StringField(2)