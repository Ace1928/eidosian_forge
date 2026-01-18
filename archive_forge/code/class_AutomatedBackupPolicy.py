from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AutomatedBackupPolicy(_messages.Message):
    """Defines an automated backup policy for a table

  Fields:
    frequency: Required. How frequently automated backups should occur. The
      only supported value at this time is 24 hours.
    retentionPeriod: Required. How long the automated backups should be
      retained. The only supported value at this time is 3 days.
  """
    frequency = _messages.StringField(1)
    retentionPeriod = _messages.StringField(2)