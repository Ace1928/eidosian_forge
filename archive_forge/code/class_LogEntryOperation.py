from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LogEntryOperation(_messages.Message):
    """Additional information about a potentially long-running operation with
  which a log entry is associated.

  Fields:
    first: Optional. Set this to True if this is the first log entry in the
      operation.
    id: Optional. An arbitrary operation identifier. Log entries with the same
      identifier are assumed to be part of the same operation.
    last: Optional. Set this to True if this is the last log entry in the
      operation.
    producer: Optional. An arbitrary producer identifier. The combination of
      `id` and `producer` must be globally unique. Examples for `producer`:
      `"MyDivision.MyBigCompany.com"`, `"github.com/MyProject/MyApplication"`.
  """
    first = _messages.BooleanField(1)
    id = _messages.StringField(2)
    last = _messages.BooleanField(3)
    producer = _messages.StringField(4)