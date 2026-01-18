from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EmailPreferences(_messages.Message):
    """Represents preferences for sending email notifications for transfer run
  events.

  Fields:
    enableFailureEmail: If true, email notifications will be sent on transfer
      run failures.
  """
    enableFailureEmail = _messages.BooleanField(1)