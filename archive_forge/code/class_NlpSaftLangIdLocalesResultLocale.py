from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NlpSaftLangIdLocalesResultLocale(_messages.Message):
    """A NlpSaftLangIdLocalesResultLocale object.

  Fields:
    languageCode: A BCP 47 language code that includes region information. For
      example, "pt-BR" or "pt-PT". This field will always be populated.
  """
    languageCode = _messages.StringField(1)