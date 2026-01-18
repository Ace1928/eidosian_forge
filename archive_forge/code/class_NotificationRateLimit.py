from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NotificationRateLimit(_messages.Message):
    """Control over the rate of notifications sent to this alert policy's
  notification channels.

  Fields:
    period: Not more than one notification per period.
  """
    period = _messages.StringField(1)