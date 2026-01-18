from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class StackTraceElement(_messages.Message):
    """A single stack element (frame) where an error occurred.

  Fields:
    position: The source position information of the stack trace element.
    routine: The routine where the error occurred.
    step: The step the error occurred at.
  """
    position = _messages.MessageField('Position', 1)
    routine = _messages.StringField(2)
    step = _messages.StringField(3)