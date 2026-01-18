from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SupportedLanguage(_messages.Message):
    """A single supported language response corresponds to information related
  to one supported language.

  Fields:
    displayName: Human readable name of the language localized in the display
      language specified in the request.
    languageCode: Supported language code, generally consisting of its ISO
      639-1 identifier, for example, 'en', 'ja'. In certain cases, BCP-47
      codes including language and region identifiers are returned (for
      example, 'zh-TW' and 'zh-CN')
    supportSource: Can be used as source language.
    supportTarget: Can be used as target language.
  """
    displayName = _messages.StringField(1)
    languageCode = _messages.StringField(2)
    supportSource = _messages.BooleanField(3)
    supportTarget = _messages.BooleanField(4)