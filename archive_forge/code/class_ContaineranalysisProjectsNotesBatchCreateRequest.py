from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ContaineranalysisProjectsNotesBatchCreateRequest(_messages.Message):
    """A ContaineranalysisProjectsNotesBatchCreateRequest object.

  Fields:
    batchCreateNotesRequest: A BatchCreateNotesRequest resource to be passed
      as the request body.
    parent: Required. The name of the project in the form of
      `projects/[PROJECT_ID]`, under which the notes are to be created.
  """
    batchCreateNotesRequest = _messages.MessageField('BatchCreateNotesRequest', 1)
    parent = _messages.StringField(2, required=True)