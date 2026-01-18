from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class JsonOptions(_messages.Message):
    """Json Options for load and make external tables.

  Fields:
    encoding: Optional. The character encoding of the data. The supported
      values are UTF-8, UTF-16BE, UTF-16LE, UTF-32BE, and UTF-32LE. The
      default value is UTF-8.
  """
    encoding = _messages.StringField(1)