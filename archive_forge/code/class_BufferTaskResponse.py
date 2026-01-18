from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BufferTaskResponse(_messages.Message):
    """Response message for BufferTask.

  Fields:
    task: The created task.
  """
    task = _messages.MessageField('Task', 1)