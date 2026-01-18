from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UpcomingMaintenanceTimeWindow(_messages.Message):
    """Represents a window of time using two timestamps: `earliest` and
  `latest`.

  Fields:
    earliest: A string attribute.
    latest: A string attribute.
  """
    earliest = _messages.StringField(1)
    latest = _messages.StringField(2)