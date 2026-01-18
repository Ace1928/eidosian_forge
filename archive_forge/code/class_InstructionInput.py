from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InstructionInput(_messages.Message):
    """An input of an instruction, as a reference to an output of a producer
  instruction.

  Fields:
    outputNum: The output index (origin zero) within the producer.
    producerInstructionIndex: The index (origin zero) of the parallel
      instruction that produces the output to be consumed by this input. This
      index is relative to the list of instructions in this input's
      instruction's containing MapTask.
  """
    outputNum = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    producerInstructionIndex = _messages.IntegerField(2, variant=_messages.Variant.INT32)