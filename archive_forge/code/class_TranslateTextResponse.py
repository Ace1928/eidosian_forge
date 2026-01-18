from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TranslateTextResponse(_messages.Message):
    """A TranslateTextResponse object.

  Fields:
    glossaryTranslations: Text translation responses if a glossary is provided
      in the request. This can be the same as `translations` if no terms
      apply. This field has the same length as `contents`.
    translations: Text translation responses with no glossary applied. This
      field has the same length as `contents`.
  """
    glossaryTranslations = _messages.MessageField('Translation', 1, repeated=True)
    translations = _messages.MessageField('Translation', 2, repeated=True)