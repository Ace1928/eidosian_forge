from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UserLanguage(_messages.Message):
    """JSON template for a language entry.

  Fields:
    customLanguage: Other language. User can provide own language name if
      there is no corresponding Google III language code. If this is set
      LanguageCode can't be set
    languageCode: Language Code. Should be used for storing Google III
      LanguageCode string representation for language. Illegal values cause
      SchemaException.
  """
    customLanguage = _messages.StringField(1)
    languageCode = _messages.StringField(2)