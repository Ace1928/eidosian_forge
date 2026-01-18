from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DumpFlag(_messages.Message):
    """Dump flag definition.

  Fields:
    name: The name of the flag
    value: The value of the flag.
  """
    name = _messages.StringField(1)
    value = _messages.StringField(2)