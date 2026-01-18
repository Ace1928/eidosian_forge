from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UpdatePhraseSetRequest(_messages.Message):
    """Request message for the UpdatePhraseSet method.

  Fields:
    phraseSet: Required. The PhraseSet to update. The PhraseSet's `name` field
      is used to identify the PhraseSet to update. Format:
      `projects/{project}/locations/{location}/phraseSets/{phrase_set}`.
    updateMask: The list of fields to update. If empty, all non-default valued
      fields are considered for update. Use `*` to update the entire PhraseSet
      resource.
    validateOnly: If set, validate the request and preview the updated
      PhraseSet, but do not actually update it.
  """
    phraseSet = _messages.MessageField('PhraseSet', 1)
    updateMask = _messages.StringField(2)
    validateOnly = _messages.BooleanField(3)