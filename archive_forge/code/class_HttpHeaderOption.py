from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HttpHeaderOption(_messages.Message):
    """Specification determining how headers are added to requests or
  responses.

  Fields:
    headerName: The name of the header.
    headerValue: The value of the header to add.
    replace: If false, headerValue is appended to any values that already
      exist for the header. If true, headerValue is set for the header,
      discarding any values that were set for that header. The default value
      is false.
  """
    headerName = _messages.StringField(1)
    headerValue = _messages.StringField(2)
    replace = _messages.BooleanField(3)