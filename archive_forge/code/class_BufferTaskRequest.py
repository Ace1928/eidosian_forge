from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BufferTaskRequest(_messages.Message):
    """Request message for BufferTask.

  Fields:
    body: Optional. Body of the HTTP request. The body can take any generic
      value. The value is written to the HttpRequest of the [Task].
  """
    body = _messages.MessageField('HttpBody', 1)