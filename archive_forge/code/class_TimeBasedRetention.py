from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TimeBasedRetention(_messages.Message):
    """A time based retention policy specifies that all backups within a
  certain time period should be retained.

  Fields:
    retentionPeriod: The retention period.
  """
    retentionPeriod = _messages.StringField(1)