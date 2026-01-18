from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CharacterMaskConfig(_messages.Message):
    """Masks a string by replacing its characters with a fixed character.

  Fields:
    maskingCharacter: Character to mask the sensitive values. If not supplied,
      defaults to "*".
  """
    maskingCharacter = _messages.StringField(1)