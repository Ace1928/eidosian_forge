from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigatewayApiConfigFile(_messages.Message):
    """A lightweight description of a file.

  Fields:
    contents: The bytes that constitute the file.
    path: The file path (full or relative path). This is typically the path of
      the file when it is uploaded.
  """
    contents = _messages.BytesField(1)
    path = _messages.StringField(2)