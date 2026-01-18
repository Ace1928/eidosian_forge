from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DumpFlags(_messages.Message):
    """Dump flags definition.

  Fields:
    dumpFlags: The flags for the initial dump.
  """
    dumpFlags = _messages.MessageField('DumpFlag', 1, repeated=True)