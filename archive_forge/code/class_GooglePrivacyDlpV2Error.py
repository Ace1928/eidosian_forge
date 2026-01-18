from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2Error(_messages.Message):
    """Details information about an error encountered during job execution or
  the results of an unsuccessful activation of the JobTrigger.

  Fields:
    details: Detailed error codes and messages.
    timestamps: The times the error occurred. List includes the oldest
      timestamp and the last 9 timestamps.
  """
    details = _messages.MessageField('GoogleRpcStatus', 1)
    timestamps = _messages.StringField(2, repeated=True)