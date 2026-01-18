from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RenameInstanceRequest(_messages.Message):
    """Message requesting rename of a server.

  Fields:
    newInstanceId: Required. The new `id` of the instance.
  """
    newInstanceId = _messages.StringField(1)