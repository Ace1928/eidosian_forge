from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RomanizeTextResponse(_messages.Message):
    """The response message for synchronous romanization.

  Fields:
    romanizations: Text romanization responses. This field has the same length
      as `contents`.
  """
    romanizations = _messages.MessageField('Romanization', 1, repeated=True)