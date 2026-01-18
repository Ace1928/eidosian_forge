from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AudioMapping(_messages.Message):
    """The mapping for the JobConfig.edit_list atoms with audio
  EditAtom.inputs.

  Fields:
    atomKey: Required. The EditAtom.key that references the atom with audio
      inputs in the JobConfig.edit_list.
    gainDb: Audio volume control in dB. Negative values decrease volume,
      positive values increase. The default is 0.
    inputChannel: Required. The zero-based index of the channel in the input
      audio stream.
    inputKey: Required. The Input.key that identifies the input file.
    inputTrack: Required. The zero-based index of the track in the input file.
    outputChannel: Required. The zero-based index of the channel in the output
      audio stream.
  """
    atomKey = _messages.StringField(1)
    gainDb = _messages.FloatField(2)
    inputChannel = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    inputKey = _messages.StringField(4)
    inputTrack = _messages.IntegerField(5, variant=_messages.Variant.INT32)
    outputChannel = _messages.IntegerField(6, variant=_messages.Variant.INT32)