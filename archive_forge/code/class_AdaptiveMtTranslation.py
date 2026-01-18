from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AdaptiveMtTranslation(_messages.Message):
    """An AdaptiveMt translation.

  Fields:
    translatedText: Output only. The translated text.
  """
    translatedText = _messages.StringField(1)