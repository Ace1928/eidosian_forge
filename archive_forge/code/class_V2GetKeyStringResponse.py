from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class V2GetKeyStringResponse(_messages.Message):
    """Response message for `GetKeyString` method.

  Fields:
    keyString: An encrypted and signed value of the key.
  """
    keyString = _messages.StringField(1)