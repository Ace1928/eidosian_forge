from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SpeechAdaptation(_messages.Message):
    """Provides "hints" to the speech recognizer to favor specific words and
  phrases in the results. PhraseSets can be specified as an inline resource,
  or a reference to an existing PhraseSet resource.

  Fields:
    customClasses: A list of inline CustomClasses. Existing CustomClass
      resources can be referenced directly in a PhraseSet.
    phraseSets: A list of inline or referenced PhraseSets.
  """
    customClasses = _messages.MessageField('CustomClass', 1, repeated=True)
    phraseSets = _messages.MessageField('AdaptationPhraseSet', 2, repeated=True)