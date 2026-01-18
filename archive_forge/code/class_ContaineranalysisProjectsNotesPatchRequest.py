from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ContaineranalysisProjectsNotesPatchRequest(_messages.Message):
    """A ContaineranalysisProjectsNotesPatchRequest object.

  Fields:
    name: Required. The name of the note in the form of
      `projects/[PROVIDER_ID]/notes/[NOTE_ID]`.
    note: A Note resource to be passed as the request body.
    updateMask: The fields to update.
  """
    name = _messages.StringField(1, required=True)
    note = _messages.MessageField('Note', 2)
    updateMask = _messages.StringField(3)