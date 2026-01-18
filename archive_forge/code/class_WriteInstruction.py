from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class WriteInstruction(_messages.Message):
    """An instruction that writes records. Takes one input, produces no
  outputs.

  Fields:
    input: The input.
    sink: The sink to write to.
  """
    input = _messages.MessageField('InstructionInput', 1)
    sink = _messages.MessageField('Sink', 2)