from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FileInputSource(_messages.Message):
    """An inlined file.

  Fields:
    content: Required. The file's byte contents.
    displayName: Required. The file's display name.
    mimeType: Required. The file's mime type.
  """
    content = _messages.BytesField(1)
    displayName = _messages.StringField(2)
    mimeType = _messages.StringField(3)