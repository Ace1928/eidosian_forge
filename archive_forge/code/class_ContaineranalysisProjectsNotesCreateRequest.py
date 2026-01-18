from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ContaineranalysisProjectsNotesCreateRequest(_messages.Message):
    """A ContaineranalysisProjectsNotesCreateRequest object.

  Fields:
    note: A Note resource to be passed as the request body.
    noteId: Required. The ID to use for this note.
    parent: Required. The name of the project in the form of
      `projects/[PROJECT_ID]`, under which the note is to be created.
  """
    note = _messages.MessageField('Note', 1)
    noteId = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)