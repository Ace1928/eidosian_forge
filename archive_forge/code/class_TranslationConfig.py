from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TranslationConfig(_messages.Message):
    """Translation configuration. Use to translate the given audio into text
  for the desired language.

  Fields:
    targetLanguage: Required. The language code to translate to.
  """
    targetLanguage = _messages.StringField(1)