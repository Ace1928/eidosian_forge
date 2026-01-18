from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CEL(_messages.Message):
    """Filters in CEL.

  Fields:
    cel: The filter logic in CEL.
    notification: A notification sent back to SCM if the cel program fails.
  """
    cel = _messages.StringField(1)
    notification = _messages.StringField(2)