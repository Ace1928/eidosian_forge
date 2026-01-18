from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ToolExecutionStep(_messages.Message):
    """Generic tool step to be used for binaries we do not explicitly support.
  For example: running cp to copy artifacts from one location to another.

  Fields:
    toolExecution: A Tool execution. - In response: present if set by
      create/update request - In create/update request: optional
  """
    toolExecution = _messages.MessageField('ToolExecution', 1)