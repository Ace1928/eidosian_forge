from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleFirestoreAdminV1DailyRecurrence(_messages.Message):
    """Represents a recurring schedule that runs at a specific time every day.
  The time zone is UTC.
  """