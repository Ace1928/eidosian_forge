from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SpeechProjectsLocationsPhraseSetsPatchRequest(_messages.Message):
    """A SpeechProjectsLocationsPhraseSetsPatchRequest object.

  Fields:
    name: Output only. Identifier. The resource name of the PhraseSet. Format:
      `projects/{project}/locations/{location}/phraseSets/{phrase_set}`.
    phraseSet: A PhraseSet resource to be passed as the request body.
    updateMask: The list of fields to update. If empty, all non-default valued
      fields are considered for update. Use `*` to update the entire PhraseSet
      resource.
    validateOnly: If set, validate the request and preview the updated
      PhraseSet, but do not actually update it.
  """
    name = _messages.StringField(1, required=True)
    phraseSet = _messages.MessageField('PhraseSet', 2)
    updateMask = _messages.StringField(3)
    validateOnly = _messages.BooleanField(4)