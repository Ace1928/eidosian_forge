from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ContaineranalysisProvidersNotesPatchRequest(_messages.Message):
    """A ContaineranalysisProvidersNotesPatchRequest object.

  Fields:
    name: The name of the note. Should be of the form
      "projects/{provider_id}/notes/{note_id}".
    note: A Note resource to be passed as the request body.
    updateMask: The fields to update.
  """
    name = _messages.StringField(1, required=True)
    note = _messages.MessageField('Note', 2)
    updateMask = _messages.StringField(3)