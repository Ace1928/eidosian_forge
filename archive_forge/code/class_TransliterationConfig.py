from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TransliterationConfig(_messages.Message):
    """Configures transliteration feature on top of translation.

  Fields:
    enableTransliteration: If true, source text in romanized form can be
      translated to the target language.
  """
    enableTransliteration = _messages.BooleanField(1)