from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class SyncFlags(_messages.Message):
    """Initial sync flags for certain Cloud SQL APIs. Currently used for the
  MySQL external server initial dump.

  Fields:
    name: The name of the flag.
    value: The value of the flag. This field must be omitted if the flag
      doesn't take a value.
  """
    name = _messages.StringField(1)
    value = _messages.StringField(2)