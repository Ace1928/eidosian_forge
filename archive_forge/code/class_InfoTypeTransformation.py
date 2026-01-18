from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InfoTypeTransformation(_messages.Message):
    """A transformation to apply to text that is identified as a specific
  info_type.

  Fields:
    characterMaskConfig: Config for character mask.
    cryptoHashConfig: Config for crypto hash.
    dateShiftConfig: Config for date shift.
    infoTypes: `InfoTypes` to apply this transformation to. If this is not
      specified, this transformation becomes the default transformation, and
      is used for any `info_type` that is not specified in another
      transformation.
    redactConfig: Config for text redaction.
    replaceWithInfoTypeConfig: Config for replace with InfoType.
  """
    characterMaskConfig = _messages.MessageField('CharacterMaskConfig', 1)
    cryptoHashConfig = _messages.MessageField('CryptoHashConfig', 2)
    dateShiftConfig = _messages.MessageField('DateShiftConfig', 3)
    infoTypes = _messages.StringField(4, repeated=True)
    redactConfig = _messages.MessageField('RedactConfig', 5)
    replaceWithInfoTypeConfig = _messages.MessageField('ReplaceWithInfoTypeConfig', 6)