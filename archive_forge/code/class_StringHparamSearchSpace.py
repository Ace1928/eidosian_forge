from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StringHparamSearchSpace(_messages.Message):
    """Search space for string and enum.

  Fields:
    candidates: Canididates for the string or enum parameter in lower case.
  """
    candidates = _messages.StringField(1, repeated=True)