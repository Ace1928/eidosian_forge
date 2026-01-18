from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SpeechProjectsLocationsPhraseSetsCreateRequest(_messages.Message):
    """A SpeechProjectsLocationsPhraseSetsCreateRequest object.

  Fields:
    parent: Required. The project and location where this PhraseSet will be
      created. The expected format is
      `projects/{project}/locations/{location}`.
    phraseSet: A PhraseSet resource to be passed as the request body.
    phraseSetId: The ID to use for the PhraseSet, which will become the final
      component of the PhraseSet's resource name. This value should be 4-63
      characters, and valid characters are /a-z-/.
    validateOnly: If set, validate the request and preview the PhraseSet, but
      do not actually create it.
  """
    parent = _messages.StringField(1, required=True)
    phraseSet = _messages.MessageField('PhraseSet', 2)
    phraseSetId = _messages.StringField(3)
    validateOnly = _messages.BooleanField(4)