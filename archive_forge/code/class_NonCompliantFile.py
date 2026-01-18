from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NonCompliantFile(_messages.Message):
    """Details about files that caused a compliance check to fail.
  display_command is a single command that can be used to display a list of
  non compliant files. When there is no such command, we can also iterate a
  list of non compliant file using 'path'.

  Fields:
    displayCommand: Command to display the non-compliant files.
    path: Empty if `display_command` is set.
    reason: Explains why a file is non compliant for a CIS check.
  """
    displayCommand = _messages.StringField(1)
    path = _messages.StringField(2)
    reason = _messages.StringField(3)