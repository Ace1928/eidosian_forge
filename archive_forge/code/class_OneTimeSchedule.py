from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class OneTimeSchedule(_messages.Message):
    """Sets the time for a one time patch deployment. Timestamp is in
  [RFC3339](https://www.ietf.org/rfc/rfc3339.txt) text format.

  Fields:
    executeTime: Required. The desired patch job execution time.
  """
    executeTime = _messages.StringField(1)