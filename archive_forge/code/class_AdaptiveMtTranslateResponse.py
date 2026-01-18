from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AdaptiveMtTranslateResponse(_messages.Message):
    """An AdaptiveMtTranslate response.

  Fields:
    languageCode: Output only. The translation's language code.
    translations: Output only. The translation.
  """
    languageCode = _messages.StringField(1)
    translations = _messages.MessageField('AdaptiveMtTranslation', 2, repeated=True)