from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LearningGenaiRootLanguageFilterResult(_messages.Message):
    """A LearningGenaiRootLanguageFilterResult object.

  Fields:
    allowed: False when query or response should be filtered out due to
      unsupported language.
    detectedLanguage: Language of the query or response.
    detectedLanguageProbability: Probability of the language predicted as
      returned by LangID.
  """
    allowed = _messages.BooleanField(1)
    detectedLanguage = _messages.StringField(2)
    detectedLanguageProbability = _messages.FloatField(3, variant=_messages.Variant.FLOAT)