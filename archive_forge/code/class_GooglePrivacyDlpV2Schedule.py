from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2Schedule(_messages.Message):
    """Schedule for inspect job triggers.

  Fields:
    recurrencePeriodDuration: With this option a job is started on a regular
      periodic basis. For example: every day (86400 seconds). A scheduled
      start time will be skipped if the previous execution has not ended when
      its scheduled time occurs. This value must be set to a time duration
      greater than or equal to 1 day and can be no longer than 60 days.
  """
    recurrencePeriodDuration = _messages.StringField(1)