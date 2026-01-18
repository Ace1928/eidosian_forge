from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TextMapping(_messages.Message):
    """The mapping for the JobConfig.edit_list atoms with text EditAtom.inputs.

  Fields:
    atomKey: Required. The EditAtom.key that references atom with text inputs
      in the JobConfig.edit_list.
    inputKey: Required. The Input.key that identifies the input file.
    inputTrack: Required. The zero-based index of the track in the input file.
  """
    atomKey = _messages.StringField(1)
    inputKey = _messages.StringField(2)
    inputTrack = _messages.IntegerField(3, variant=_messages.Variant.INT32)